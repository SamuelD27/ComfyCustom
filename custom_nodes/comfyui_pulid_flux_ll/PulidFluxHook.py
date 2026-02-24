import torch
from einops import rearrange
from torch import Tensor
from comfy.ldm.flux.layers import timestep_embedding
import comfy
from .patch_util import PatchKeys

def set_model_dit_patch_replace(model, patch_kwargs, key, is_last_double=False):
    to = model.model_options["transformer_options"]
    if "patches_replace" not in to:
        to["patches_replace"] = {}
    if "dit" not in to["patches_replace"]:
        to["patches_replace"]["dit"] = {}

    if key not in to["patches_replace"]["dit"]:
        if "double_block" in key:
            if is_last_double:
                to["patches_replace"]["dit"][key] = LastDitDoubleBlockReplace(pulid_patch, **patch_kwargs)
            else:
                to["patches_replace"]["dit"][key] = DitDoubleBlockReplace(pulid_patch, **patch_kwargs)
        else:
            to["patches_replace"]["dit"][key] = DitSingleBlockReplace(pulid_patch, **patch_kwargs)
    else:
        to["patches_replace"]["dit"][key].add(pulid_patch, **patch_kwargs)

PULID_DIM = 3072

def pulid_patch(img, pulid_model=None, ca_idx=None, weight=1.0, embedding=None, mask=None, transformer_options={}):
    model_dim = img.shape[-1]
    ca = pulid_model.model.pulid_ca[ca_idx].to(img.device)
    if embedding.device != img.device:
        embedding = embedding.to(device=img.device, dtype=img.dtype)

    if model_dim == PULID_DIM:
        pulid_img = weight * ca(embedding, img)
    else:
        if model_dim % PULID_DIM != 0:
            raise ValueError(
                f"PuLID dimension adapter requires model_dim ({model_dim}) "
                f"to be a multiple of PULID_DIM ({PULID_DIM})"
            )
        factor = model_dim // PULID_DIM
        img_proj = img.reshape(*img.shape[:-1], factor, PULID_DIM).mean(dim=-2)
        ca_out = weight * ca(embedding, img_proj)
        pulid_img = ca_out.unsqueeze(-2).expand(*ca_out.shape[:-1], factor, PULID_DIM).reshape(*img.shape)

    if mask is not None:
        pulid_temp_attrs = transformer_options.get(PatchKeys.pulid_patch_key_attrs, {})
        latent_image_shape = pulid_temp_attrs.get("latent_image_shape", None)
        if latent_image_shape is not None:
            bs, c, h, w = latent_image_shape
            mask = comfy.sampler_helpers.prepare_mask(mask, (bs, c, h, w), img.device)
            patch_size = transformer_options[PatchKeys.running_net_model].patch_size
            mask = comfy.ldm.common_dit.pad_to_patch_size(mask, (patch_size, patch_size))
            mask = rearrange(mask, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch_size, pw=patch_size)
            mask = mask[..., 0].unsqueeze(-1).repeat(1, 1, pulid_img.shape[-1]).to(dtype=pulid_img.dtype)
            del patch_size, latent_image_shape

        pulid_img = pulid_img * mask
        del mask, pulid_temp_attrs

    return pulid_img

class DitDoubleBlockReplace:
    def __init__(self, callback, **kwargs):
        self.callback = [callback]
        self.kwargs = [kwargs]

    def add(self, callback, **kwargs):
        self.callback.append(callback)
        self.kwargs.append(kwargs)

        for key, value in kwargs.items():
            setattr(self, key, value)

    def __call__(self, input_args, extra_options):
        transformer_options = extra_options["transformer_options"]
        pulid_temp_attrs = transformer_options.get(PatchKeys.pulid_patch_key_attrs, {})
        sigma = pulid_temp_attrs["timesteps"][0].detach().cpu().item()
        out = extra_options["original_block"](input_args)
        img = out['img']
        temp_img = img
        for i, callback in enumerate(self.callback):
            if self.kwargs[i]["sigma_start"] >= sigma >= self.kwargs[i]["sigma_end"]:
                img = img + callback(temp_img,
                                     pulid_model=self.kwargs[i]['pulid_model'],
                                     ca_idx=self.kwargs[i]['ca_idx'],
                                     weight=self.kwargs[i]['weight'],
                                     embedding=self.kwargs[i]['embedding'],
                                     mask = self.kwargs[i]['mask'],
                                     transformer_options=transformer_options
                                     )
        out['img'] = img

        del temp_img, pulid_temp_attrs, sigma, transformer_options, img

        return out


class LastDitDoubleBlockReplace(DitDoubleBlockReplace):
    def __call__(self, input_args, extra_options):
        out = super().__call__(input_args, extra_options)
        transformer_options = extra_options["transformer_options"]
        pulid_temp_attrs = transformer_options.get(PatchKeys.pulid_patch_key_attrs, {})
        pulid_temp_attrs["double_blocks_txt"] = out['txt']
        return out

class DitSingleBlockReplace:
    def __init__(self, callback, **kwargs):
        self.callback = [callback]
        self.kwargs = [kwargs]

    def add(self, callback, **kwargs):
        self.callback.append(callback)
        self.kwargs.append(kwargs)

        for key, value in kwargs.items():
            setattr(self, key, value)

    def __call__(self, input_args, extra_options):
        transformer_options = extra_options["transformer_options"]
        pulid_temp_attrs = transformer_options.get(PatchKeys.pulid_patch_key_attrs, {})

        out = extra_options["original_block"](input_args)

        sigma = pulid_temp_attrs["timesteps"][0].detach().cpu().item()
        img = out['img']
        txt = pulid_temp_attrs['double_blocks_txt']
        real_img, txt = img[:, txt.shape[1]:, ...], img[:, :txt.shape[1], ...]
        temp_img = real_img
        for i, callback in enumerate(self.callback):
            if self.kwargs[i]["sigma_start"] >= sigma >= self.kwargs[i]["sigma_end"]:
                real_img = real_img + callback(temp_img,
                                               pulid_model=self.kwargs[i]['pulid_model'],
                                               ca_idx=self.kwargs[i]['ca_idx'],
                                               weight=self.kwargs[i]['weight'],
                                               embedding=self.kwargs[i]['embedding'],
                                               mask=self.kwargs[i]['mask'],
                                               transformer_options = transformer_options,
                                               )

        img = torch.cat((txt, real_img), 1)

        out['img'] = img

        del temp_img, pulid_temp_attrs, sigma, transformer_options, real_img, img

        return out

def pulid_forward_orig(
    self,
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    timesteps: Tensor,
    y: Tensor,
    guidance: Tensor = None,
    control=None,
    transformer_options={},
    attn_mask: Tensor = None,
) -> Tensor:
    patches = transformer_options.get("patches", {})
    patches_replace = transformer_options.get("patches_replace", {})

    if img.ndim != 3 or txt.ndim != 3:
        raise ValueError("Input img and txt tensors must have 3 dimensions.")

    transformer_options[PatchKeys.running_net_model] = self

    # running on sequences img
    img = self.img_in(img)
    vec = self.time_in(timestep_embedding(timesteps, 256).to(img.dtype))
    if self.params.guidance_embed:
        if guidance is not None:
            vec = vec + self.guidance_in(timestep_embedding(guidance, 256).to(img.dtype))

    if self.vector_in is not None:
        if y is None:
            y = torch.zeros((img.shape[0], self.params.vec_in_dim), device=img.device, dtype=img.dtype)
        vec = vec + self.vector_in(y[:, :self.params.vec_in_dim])

    if self.txt_norm is not None:
        txt = self.txt_norm(txt)
    txt = self.txt_in(txt)

    # Flux 2 global modulation
    vec_orig = vec
    if self.params.global_modulation:
        vec = (self.double_stream_modulation_img(vec_orig), self.double_stream_modulation_txt(vec_orig))

    if "post_input" in patches:
        for p in patches["post_input"]:
            out = p({"img": img, "txt": txt, "img_ids": img_ids, "txt_ids": txt_ids})
            img = out["img"]
            txt = out["txt"]
            img_ids = out["img_ids"]
            txt_ids = out["txt_ids"]

    if img_ids is not None:
        ids = torch.cat((txt_ids, img_ids), dim=1)
        pe = self.pe_embedder(ids)
    else:
        pe = None

    blocks_replace = patches_replace.get("dit", {})

    transformer_options["total_blocks"] = len(self.double_blocks)
    transformer_options["block_type"] = "double"
    for i, block in enumerate(self.double_blocks):
        transformer_options["block_index"] = i
        if ("double_block", i) in blocks_replace:
            def block_wrap(args):
                out = {}
                out["img"], out["txt"] = block(img=args["img"],
                                               txt=args["txt"],
                                               vec=args["vec"],
                                               pe=args["pe"],
                                               attn_mask=args.get("attn_mask"),
                                               transformer_options=args.get("transformer_options"))
                return out
            out = blocks_replace[("double_block", i)]({"img": img,
                                                       "txt": txt,
                                                       "vec": vec,
                                                       "pe": pe,
                                                       "attn_mask": attn_mask,
                                                       "transformer_options": transformer_options},
                                                      {"original_block": block_wrap,
                                                       "transformer_options": transformer_options})
            txt = out["txt"]
            img = out["img"]
        else:
            img, txt = block(img=img, txt=txt, vec=vec, pe=pe, attn_mask=attn_mask, transformer_options=transformer_options)

        if control is not None:
            control_i = control.get("input")
            if i < len(control_i):
                add = control_i[i]
                if add is not None:
                    img[:, :add.shape[1]] += add

    if img.dtype == torch.float16:
        img = torch.nan_to_num(img, nan=0.0, posinf=65504, neginf=-65504)

    img = torch.cat((txt, img), 1)

    # Flux 2 single stream modulation
    if self.params.global_modulation:
        vec, _ = self.single_stream_modulation(vec_orig)

    transformer_options["total_blocks"] = len(self.single_blocks)
    transformer_options["block_type"] = "single"
    for i, block in enumerate(self.single_blocks):
        transformer_options["block_index"] = i
        if ("single_block", i) in blocks_replace:
            def block_wrap(args):
                out = {}
                out["img"] = block(args["img"],
                                   vec=args["vec"],
                                   pe=args["pe"],
                                   attn_mask=args.get("attn_mask"),
                                   transformer_options=args.get("transformer_options"))
                return out
            out = blocks_replace[("single_block", i)]({"img": img,
                                                       "vec": vec,
                                                       "pe": pe,
                                                       "attn_mask": attn_mask,
                                                       "transformer_options": transformer_options},
                                                      {"original_block": block_wrap,
                                                       "transformer_options": transformer_options})
            img = out["img"]
        else:
            img = block(img, vec=vec, pe=pe, attn_mask=attn_mask, transformer_options=transformer_options)

        if control is not None:
            control_o = control.get("output")
            if i < len(control_o):
                add = control_o[i]
                if add is not None:
                    img[:, txt.shape[1] : txt.shape[1] + add.shape[1], ...] += add

    img = img[:, txt.shape[1]:, ...]

    img = self.final_layer(img, vec_orig)

    del transformer_options[PatchKeys.running_net_model]

    return img


def pulid_enter(img, img_ids, txt, txt_ids, timesteps, y, guidance, control, attn_mask, transformer_options):
    pulid_temp_attrs = transformer_options.get(PatchKeys.pulid_patch_key_attrs, {})
    pulid_temp_attrs['timesteps'] = timesteps
    return img, img_ids, txt, txt_ids, timesteps, y, guidance, control, attn_mask


def pulid_patch_double_blocks_after(img, txt, transformer_options):
    pulid_temp_attrs = transformer_options.get(PatchKeys.pulid_patch_key_attrs, {})
    pulid_temp_attrs['double_blocks_txt'] = txt
    return img, txt
