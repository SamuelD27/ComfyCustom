#!/usr/bin/env python3
"""Generate ComfyUI workflow JSON files.

Usage:
    .venv/bin/python scripts/build_workflows.py
"""

from pathlib import Path

from workflow_builder import make_workflow, save_workflow

PROJECT_ROOT = Path(__file__).resolve().parent.parent

QIE_INSTRUCTION = (
    "Describe the key features of the input image (facial structure, body "
    "proportions, skin tone, distinctive features), then explain how the "
    "user's text instruction modifies the scene while preserving the person's "
    "identity. Generate a new image that matches the user's requirements while "
    "maintaining complete facial and body consistency with the original input."
)


def _load_prompts_text():
    """Load diversity prompts from file, or return empty string."""
    prompts_path = PROJECT_ROOT / "user/default/workflows/prompts/dataset_diversity_prompts.txt"
    if prompts_path.exists():
        return prompts_path.read_text(encoding="utf-8").rstrip("\n")
    return ""


def build_dataset_gen():
    """Build the QIE2511 dataset generation workflow."""

    prompts_text = _load_prompts_text()

    # ---------------------------------------------------------------
    # Node definitions
    # ---------------------------------------------------------------
    nodes_def = [
        # -- Model Loaders (x=100) --
        {
            "id": 1,
            "type": "UNETLoader",
            "title": "Load QIE2511 Diffusion Model",
            "pos": [100, 100],
            "size": [315, 82],
            "inputs": [],
            "outputs": [("MODEL", "MODEL")],
            "widgets_values": [
                "qwen_image_edit_2511_bf16.safetensors",
                "default",
            ],
        },
        {
            "id": 2,
            "type": "LoraLoaderModelOnly",
            "title": "Lightning LoRA (Muted)",
            "pos": [100, 250],
            "size": [315, 82],
            "mode": 2,  # MUTED / bypassed
            "inputs": [
                ("model", "MODEL"),
            ],
            "outputs": [("MODEL", "MODEL")],
            "widgets_values": [
                "Qwen-Image-Edit-2511-Lightning-8steps-V1.0-fp32.safetensors",
                1.0,
            ],
        },
        {
            "id": 3,
            "type": "CLIPLoader",
            "title": "Load Qwen2.5-VL Text Encoder",
            "pos": [100, 400],
            "size": [315, 82],
            "inputs": [],
            "outputs": [("CLIP", "CLIP")],
            "widgets_values": [
                "qwen_2.5_vl_7b_fp8_scaled.safetensors",
                "qwen2_5_vl",
            ],
        },
        {
            "id": 4,
            "type": "VAELoader",
            "title": "Load QIE VAE",
            "pos": [100, 550],
            "size": [315, 82],
            "inputs": [],
            "outputs": [("VAE", "VAE")],
            "widgets_values": [
                "qwen_image_vae.safetensors",
            ],
        },

        # -- Reference Image (x=100, y=750) --
        {
            "id": 5,
            "type": "LoadImage",
            "title": "Reference Image",
            "pos": [100, 750],
            "size": [315, 314],
            "inputs": [],
            "outputs": [
                ("IMAGE", "IMAGE"),
                ("MASK", "MASK"),
            ],
            "widgets_values": ["example.png", "image"],
        },

        # -- Prompt (x=550) --
        {
            "id": 6,
            "type": "CR Prompt List",
            "title": "Diversity Prompts",
            "pos": [550, 100],
            "size": [500, 400],
            "color": "#322",
            "bgcolor": "#533",
            "inputs": [],
            "outputs": [
                ("prompt", "STRING"),
                ("body_text", "STRING"),
                ("show_help", "STRING"),
            ],
            "widgets_values": [
                "",             # prepend_text
                prompts_text,   # multiline_text
                "",             # append_text
                0,              # start_index
                1000,           # max_rows
            ],
        },
        {
            "id": 8,
            "type": "PrimitiveNode",
            "title": "Trigger Word",
            "pos": [550, 550],
            "size": [315, 82],
            "inputs": [],
            "outputs": [("STRING", "STRING")],
            "widgets_values": ["ohwx person"],
        },

        # -- QIE2511 Encoding (x=1100) --
        {
            "id": 10,
            "type": "TextEncodeQwenImageEditPlus_lrzjason",
            "title": "QIE2511 Encode (Identity)",
            "pos": [1100, 100],
            "size": [400, 350],
            "inputs": [
                ("clip", "CLIP"),           # slot 0
                ("prompt", "STRING"),       # slot 1
                ("vae", "VAE"),             # slot 2
                ("image1", "IMAGE"),        # slot 3
                ("image2", "IMAGE"),        # slot 4
                ("image3", "IMAGE"),        # slot 5
                ("image4", "IMAGE"),        # slot 6
                ("image5", "IMAGE"),        # slot 7
            ],
            "outputs": [
                ("conditioning", "CONDITIONING"),   # slot 0
                ("image1", "IMAGE"),                # slot 1
                ("image2", "IMAGE"),                # slot 2
                ("image3", "IMAGE"),                # slot 3
                ("image4", "IMAGE"),                # slot 4
                ("image5", "IMAGE"),                # slot 5
                ("latent", "LATENT"),               # slot 6
            ],
            "widgets_values": [
                True,       # enable_resize
                True,       # enable_vl_resize
                False,      # skip_first_image_resize
                "bicubic",  # upscale_method
                "center",   # crop
                QIE_INSTRUCTION,  # instruction
            ],
        },
        {
            "id": 11,
            "type": "CLIPTextEncode",
            "title": "Negative (Empty)",
            "pos": [1100, 500],
            "size": [400, 150],
            "inputs": [
                ("clip", "CLIP"),
            ],
            "outputs": [
                ("CONDITIONING", "CONDITIONING"),
            ],
            "widgets_values": [""],
        },

        # -- Sampler (x=1600) --
        {
            "id": 15,
            "type": "KSampler",
            "title": "KSampler",
            "pos": [1600, 100],
            "size": [315, 262],
            "inputs": [
                ("model", "MODEL"),             # slot 0
                ("positive", "CONDITIONING"),   # slot 1
                ("negative", "CONDITIONING"),   # slot 2
                ("latent_image", "LATENT"),     # slot 3
            ],
            "outputs": [
                ("LATENT", "LATENT"),
            ],
            "widgets_values": [
                0,          # seed
                "randomize",  # seed control
                30,         # steps
                5.0,        # cfg
                "euler",    # sampler_name
                "beta57",   # scheduler
                1.0,        # denoise
            ],
        },

        # -- Decode (x=2000) --
        {
            "id": 16,
            "type": "VAEDecode",
            "title": "VAE Decode",
            "pos": [2000, 100],
            "size": [210, 46],
            "inputs": [
                ("samples", "LATENT"),
                ("vae", "VAE"),
            ],
            "outputs": [
                ("IMAGE", "IMAGE"),
            ],
            "widgets_values": [],
        },

        # -- Captioning (x=2300) --
        {
            "id": 20,
            "type": "JC_ExtraOptions",
            "title": "JoyCaption Extra Options",
            "pos": [2300, 100],
            "size": [350, 600],
            "inputs": [],
            "outputs": [
                ("extra_options", "JOYCAPTION_EXTRA_OPTIONS"),
            ],
            "widgets_values": [
                True,   # exclude_people_info
                True,   # include_lighting
                True,   # include_camera_angle
                False,  # include_watermark
                False,  # include_JPEG_artifacts
                False,  # include_exif
                False,  # exclude_sexual
                True,   # exclude_image_resolution
                False,  # include_aesthetic_quality
                False,  # include_composition_style
                False,  # exclude_text
                False,  # specify_depth_field
                False,  # specify_lighting_sources
                False,  # do_not_use_ambiguous_language
                False,  # include_nsfw
                False,  # only_describe_most_important_elements
                False,  # do_not_include_artist_name_or_title
                False,  # identify_image_orientation
                False,  # use_vulgar_slang_and_profanity
                False,  # do_not_use_polite_euphemisms
                False,  # include_character_age
                False,  # include_camera_shot_type
                False,  # exclude_mood_feeling
                False,  # include_camera_vantage_height
                False,  # mention_watermark
                False,  # avoid_meta_descriptive_phrases
                False,  # refer_character_name
                "",     # character_name
            ],
        },
        {
            "id": 21,
            "type": "JC_adv",
            "title": "JoyCaption (Advanced)",
            "pos": [2300, 750],
            "size": [400, 350],
            "inputs": [
                ("image", "IMAGE"),                             # slot 0
                ("extra_options", "JOYCAPTION_EXTRA_OPTIONS"),  # slot 1
            ],
            "outputs": [
                ("PROMPT", "STRING"),   # slot 0
                ("STRING", "STRING"),   # slot 1
            ],
            "widgets_values": [
                "joycaption-beta-one",       # model
                "Maximum Savings (4-bit)",   # quantization
                "Straightforward",           # prompt_style / caption_type
                "medium",                    # caption_length
                512,                         # max_new_tokens
                0.6,                         # temperature
                0.9,                         # top_p
                0,                           # top_k
                "",                          # custom_prompt
                "Keep in Memory",            # memory_management
            ],
        },

        # -- Caption Assembly (x=2700) --
        {
            "id": 25,
            "type": "JoinStrings",
            "title": "Join Trigger + Caption",
            "pos": [2700, 100],
            "size": [315, 100],
            "inputs": [
                ("string1", "STRING"),   # slot 0
                ("string2", "STRING"),   # slot 1
            ],
            "outputs": [
                ("STRING", "STRING"),
            ],
            "widgets_values": [
                ", ",  # delimiter
            ],
        },

        # -- Save (x=3100) --
        {
            "id": 30,
            "type": "SaveImageKJ",
            "title": "Save Image + Caption",
            "pos": [3100, 100],
            "size": [315, 150],
            "inputs": [
                ("images", "IMAGE"),            # slot 0
                ("caption", "STRING"),          # slot 1
            ],
            "outputs": [
                ("filename", "STRING"),
            ],
            "widgets_values": [
                "img",              # filename_prefix
                "output/dataset",   # output_folder
                ".txt",             # caption_file_extension
            ],
        },
        {
            "id": 31,
            "type": "PreviewImage",
            "title": "Preview",
            "pos": [3100, 350],
            "size": [500, 500],
            "inputs": [
                ("images", "IMAGE"),
            ],
            "outputs": [],
            "widgets_values": [],
        },
    ]

    # ---------------------------------------------------------------
    # Link definitions: (src_id, src_slot, dst_id, dst_slot, type_str)
    # ---------------------------------------------------------------
    links_def = [
        # Model chain: UNETLoader -> LoRA (bypass) -> KSampler
        (1, 0, 2, 0, "MODEL"),      # UNETLoader MODEL -> LoRA model input
        (2, 0, 15, 0, "MODEL"),     # LoRA MODEL out -> KSampler model

        # CLIP to both encoders
        (3, 0, 10, 0, "CLIP"),      # CLIPLoader -> QIE encoder clip
        (3, 0, 11, 0, "CLIP"),      # CLIPLoader -> negative CLIPTextEncode clip

        # VAE to encoder and decoder
        (4, 0, 10, 2, "VAE"),       # VAELoader -> QIE encoder vae (slot 2)
        (4, 0, 16, 1, "VAE"),       # VAELoader -> VAEDecode vae

        # Reference image to encoder
        (5, 0, 10, 3, "IMAGE"),     # LoadImage IMAGE -> QIE encoder image1 (slot 3)

        # Prompt list to encoder
        (6, 0, 10, 1, "STRING"),    # CR Prompt List prompt -> QIE encoder prompt (slot 1)

        # Conditioning to KSampler
        (10, 0, 15, 1, "CONDITIONING"),  # QIE conditioning -> KSampler positive
        (10, 6, 15, 3, "LATENT"),        # QIE latent -> KSampler latent_image
        (11, 0, 15, 2, "CONDITIONING"),  # Negative conditioning -> KSampler negative

        # Sampler to decode
        (15, 0, 16, 0, "LATENT"),   # KSampler -> VAEDecode

        # Decoded image to JoyCaption
        (16, 0, 21, 0, "IMAGE"),    # VAEDecode IMAGE -> JC_adv image

        # Extra options to JoyCaption
        (20, 0, 21, 1, "JOYCAPTION_EXTRA_OPTIONS"),  # JC_ExtraOptions -> JC_adv

        # Trigger word to JoinStrings
        (8, 0, 25, 0, "STRING"),    # PrimitiveNode -> JoinStrings string1

        # Caption to JoinStrings
        (21, 1, 25, 1, "STRING"),   # JC_adv STRING output (slot 1) -> JoinStrings string2

        # Decoded image to SaveImageKJ
        (16, 0, 30, 0, "IMAGE"),    # VAEDecode IMAGE -> SaveImageKJ images

        # Joined caption to SaveImageKJ
        (25, 0, 30, 1, "STRING"),   # JoinStrings -> SaveImageKJ caption

        # Decoded image to preview
        (16, 0, 31, 0, "IMAGE"),    # VAEDecode IMAGE -> PreviewImage
    ]

    # ---------------------------------------------------------------
    # Groups
    # ---------------------------------------------------------------
    groups_def = [
        {
            "title": "Model Loaders",
            "bounding": [60, 50, 400, 630],
            "color": "#444",
        },
        {
            "title": "Reference & Prompts",
            "bounding": [60, 700, 1000, 420],
            "color": "#336",
        },
        {
            "title": "Generation",
            "bounding": [1060, 50, 900, 620],
            "color": "#363",
        },
        {
            "title": "Captioning",
            "bounding": [2260, 50, 500, 1100],
            "color": "#549",
        },
        {
            "title": "Output",
            "bounding": [3060, 50, 580, 850],
            "color": "#444",
        },
    ]

    return make_workflow(nodes_def, links_def, groups_def)


def main():
    wf = build_dataset_gen()
    out_path = PROJECT_ROOT / "user/default/workflows/dataset_gen_qie2511.json"
    save_workflow(wf, out_path)

    # Summary
    print(f"Nodes: {len(wf['nodes'])}")
    print(f"Links: {len(wf['links'])}")
    for n in wf["nodes"]:
        print(f"  {n['id']}: {n['type']} - {n.get('title', '')}")


if __name__ == "__main__":
    main()
