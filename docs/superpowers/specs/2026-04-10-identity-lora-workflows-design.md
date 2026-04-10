# Identity LoRA Workflow System Design

**Date:** 2026-04-10
**Status:** Approved
**Hardware:** ASUS GX10 / NVIDIA GB10, 120 GB usable VRAM

---

## Overview

Two ComfyUI workflows plus a helper script for identity LoRA training and inference:

1. **Dataset Generation Workflow** — generates synthetic training images from reference photos using Qwen Image Edit 2511, with integrated WD14+JoyCaption recaptioning and automatic image+caption file pair saving
2. **Final Generation Workflow** — produces high-fidelity output images using Z-Image Base with a trained identity LoRA, with multi-stage face refinement
3. **Caption Cleanup Script** — standalone Python tool for auditing and fixing captions before training

---

## Research Findings That Shaped This Design

### Z-Image Has No Zero-Shot Identity Adapters

As of April 2026, Z-Image (S3-DiT architecture) has no PuLID, InfiniteYou, IP-Adapter, or any other zero-shot identity adapter. These all exist only for Flux (MMDiT architecture). Identity preservation on Z-Image requires trained LoRAs + img2img/inpainting pipelines. This is the fundamental constraint that shapes the two-workflow system: QIE2511 generates training data, then a LoRA is trained, then Z-Image Base uses that LoRA.

### QIE2511 Is the SOTA for Synthetic Dataset Generation

Qwen Image Edit 2511 (20B MMDiT) is the established best tool for generating identity-consistent synthetic training datasets from reference photos. It takes 1-5 reference images, understands instruction-based editing ("keep this person, change the scene"), and needs no masking. The FloYo "Single Image to Character Dataset" workflow validates this approach in production.

### Captioning: Recaption Outputs, Don't Reuse Prompts

Generation prompts don't accurately describe what appeared in the output (prompt adherence is imperfect). The SOTA is to recaption generated images with WD14 + JoyCaption hybrid. JoyCaption Alpha Two has an "omit physical characteristics" mode that prevents identity features from leaking into captions, letting them bind to the trigger word instead.

### Captioning Principle: "Caption What Varies, Never What Defines Identity"

This principle remains the gold standard in April 2026. What you caption becomes promptable at inference. What you don't caption binds to the trigger word as fixed identity. Therefore: caption clothing, pose, background, lighting, expression. Do NOT caption face shape, eye color, skin tone, or other permanent identity markers.

### SAM3 Over FaceDetailer for Face Refinement

SAM3's text-grounded segmentation (Grounding DINO backend) produces tighter, more accurate face masks than YOLO-based FaceDetailer bounding boxes. With 120 GB VRAM, the extra model load is negligible.

### Z-Image Base Over Turbo for LoRA Work

Z-Image Base (undistilled S3-DiT) is better for LoRA training and inference quality. Turbo's distillation compresses the representation and reduces LoRA effectiveness to 50-70% of Base. Since speed is not a priority, Base is the correct choice.

---

## Workflow 1: Dataset Generation

### Architecture

```
LoadImage (1-5 reference photos)
    |
QwenEditConfigPreparer (to_ref=True, to_vl=True, ref_longest_edge=1024)
    |
TextEncodeQwenImageEditPlusCustom
    ← CLIPLoader (qwen_2.5_vl_7b_fp8_scaled)
    ← VAELoader (qwen_image_vae)
    ← CR Prompt List (72 curated diversity prompts)
    ← Instruction: identity-preserving system prompt
    |
UNETLoader (qwen_image_edit_2511_bf16)
    → [Optional bypass: LoraLoaderModelOnly (Lightning 8-step LoRA)]
    |
KSampler (euler, beta57, 30 steps, cfg 4-6, denoise 1.0)
    |
VAEDecode → generated_image
    |
    ├→ JoyCaption (Alpha Two, omit physical characteristics)
    |
String Concatenation: "{trigger_word}, {caption}"
    |
SaveImageKJ (image + .txt pair, same base filename)
```

### QIE2511 Configuration

- **Diffusion model:** `qwen_image_edit_2511_bf16.safetensors` (39 GB)
- **Text encoder:** `qwen_2.5_vl_7b_fp8_scaled.safetensors` (8.8 GB)
- **VAE:** `qwen_image_vae.safetensors` (0.24 GB)
- **Steps:** 30 (full quality default), 8 with Lightning LoRA toggle
- **Sampler:** euler
- **Scheduler:** beta57
- **CFG:** 4-6
- **Denoise:** 1.0 (full generation, not partial editing)
- **Resolution:** 1024x1024

### System Instruction for QIE2511

```
Describe the key features of the input image (facial structure, body proportions,
skin tone, distinctive features), then explain how the user's text instruction
modifies the scene while preserving the person's identity. Generate a new image
that matches the user's requirements while maintaining complete facial and body
consistency with the original input.
```

### Prompt Diversity Strategy

72 curated prompts using Latin hypercube sampling across 6 diversity axes:

| Axis | Categories (6-7 each) |
|---|---|
| Framing | close-up headshot, portrait, medium shot, 3/4 body, full body, environmental portrait |
| Viewpoint | frontal, 3/4 left, 3/4 right, profile left, profile right, slight overhead, slight low angle |
| Expression | neutral, slight smile, broad smile, serious, contemplative, laughing, surprised |
| Lighting | natural daylight, golden hour, studio softbox, dramatic side light, overcast diffused, backlit rim |
| Background | studio white/gray, park/garden, urban street, cafe interior, beach/waterfront, office |
| Outfit | casual, business formal, athletic, summer dress, winter layers, evening formal |

Each prompt follows the template:
```
Keep the facial features and body proportions of Picture 1 unchanged.
Generate a {framing} photograph of this person, {viewpoint}.
They are wearing {outfit}. Expression: {expression}.
Setting: {background}. Lighting: {lighting}.
High quality, photorealistic, natural skin texture, sharp focus.
```

72 prompts guarantees each category appears ~12 times with minimal combination repetition. Generate all 72, curate to best 25-40 for training.

### Captioning Pipeline

1. **JoyCaption Alpha Two** runs on the generated output image with "omit physical characteristics" mode, producing a natural language caption that describes variable attributes (clothing, pose, background, lighting) while excluding identity features
2. **String concatenation** prepends trigger word: `"{trigger_word}, {caption}"`
3. **SaveImageKJ** writes image.png + image.txt with same base filename

**Note on WD14 Tagger:** WD14 is available in the install and useful for manual quality checks, but is NOT included in the automated pipeline. JoyCaption's "omit physical characteristics" mode handles the identity-exclusion requirement directly. WD14's booru-style tags are also suboptimal for Qwen-based text encoders which prefer natural language. If the user wants WD14 tags as a secondary verification, they can add the node to preview but it does not feed into the saved caption.

### Save Logic

- **Node:** SaveImageKJ (from ComfyUI-KJNodes)
- **Output folder:** user-configurable, default `output/dataset`
- **Filename:** `{prefix}_{counter:05d}_.png` / `{prefix}_{counter:05d}_.txt`
- **Caption content:** trigger word + comma + JoyCaption output
- **Caption extension:** `.txt`

### VRAM Budget

QIE2511 (48 GB) + JoyCaption 4-bit (5 GB) + WD14 (1 GB) + VAE/CLIP (3 GB) = ~57 GB peak. Comfortable within 120 GB.

---

## Workflow 2: Final Generation

### Architecture — Four-Stage Pipeline

#### Stage 1: Base Generation

```
UNETLoader (z_image_bf16) → ModelSamplingAuraFlow (shift 3.0)
    → LoraLoaderModelOnly (identity LoRA, strength 0.75-0.85)
    → [Optional: LoraLoaderModelOnly (style LoRA)]
CLIPLoader (qwen_3_4b) → CLIPTextEncode (positive/negative)
VAELoader (z_image_ae)
EmptyZImageLatentImage (aspect ratio selection)
    |
KSampler (euler/dpmpp_2m, linear_quadratic, 20-30 steps, cfg 3-5, denoise 1.0)
    |
VAEDecode → base_image
```

#### Stage 2: Face Refinement (SAM3 + Inpaint)

```
base_image → SAM3Grounding (text: "face", confidence: 0.33)
    → face_mask
    → InpaintCropImproved (crop face region)
    → KSampler (Z-Image Base + LoRA at 0.90-1.0, 15-20 steps, denoise 0.40-0.45, cfg 3.0)
    → InpaintStitchImproved (composite back)
    → refined_image
```

#### Stage 3: Face Restoration (Optional — bypass switch (ComfyUI group mute/bypass))

```
refined_image → FaceRestoreCFNode (CodeFormer, fidelity 0.7)
    → restored_image
```

#### Stage 4: Upscale (Optional — bypass switch (ComfyUI group mute/bypass))

```
restored_image → UltimateSDUpscale
    upscale_model: 4x-UltraSharp.pth
    tile_size: 1024, denoise: 0.25-0.35
    → final_image → SaveImage
```

### Z-Image Base Configuration

- **Diffusion model:** `z_image_bf16.safetensors` (12.3 GB)
- **Text encoder:** `qwen_3_4b.safetensors` (8 GB)
- **VAE:** `z_image_ae.safetensors` (0.34 GB)
- **ModelSamplingAuraFlow shift:** 3.0
- **Sampler:** euler or dpmpp_2m
- **Scheduler:** linear_quadratic
- **Base gen steps:** 20-30
- **Base gen CFG:** 3.0-5.0
- **Face refine steps:** 15-20
- **Face refine denoise:** 0.40-0.45
- **Face refine LoRA strength:** 0.90-1.0

### Negative Prompt

```
blurry, deformed, distorted face, extra fingers, bad anatomy, watermark, text, low quality
```

### VRAM Budget

Z-Image Base (13 GB) + LoRA (0.3 GB) + SAM3 (1 GB) + text encoder (8 GB) + VAE (0.3 GB) + upscale (0.07 GB) = ~23 GB peak. Very comfortable.

---

## Caption Cleanup Script

### Location

`scripts/caption_cleanup.py`

### Usage

```bash
python scripts/caption_cleanup.py /path/to/dataset/ --trigger "ohwx woman"
```

### Features

| Flag | Description |
|---|---|
| (default) | Scan, validate trigger word, strip identity descriptors, normalize formatting |
| `--dry-run` | Show changes without modifying files |
| `--report` | Generate CSV report of all captions for manual review |
| `--retrigger "new"` | Replace existing trigger word across all captions |
| `--blocklist file.txt` | Use custom identity descriptor blocklist |

### Identity Descriptor Blocklist

Default `scripts/caption_blocklist.txt` contains:
- Facial structure: "oval face", "round face", "angular jaw", "high cheekbones", etc.
- Eyes: "brown eyes", "blue eyes", "green eyes", "almond-shaped eyes", etc.
- Skin: "fair skin", "dark complexion", "pale skin", "tanned", etc.
- Hair color: configurable — sometimes identity-defining, sometimes variable

### Operations

1. Scan for orphan files (images without captions, captions without images)
2. Validate trigger word present at start of every caption
3. Strip blocklisted identity descriptors
4. Remove hedging language: "appears to be", "seems to", "possibly", "likely"
5. Normalize formatting: consistent commas, no double spaces, clean punctuation
6. Report statistics

---

## File Structure

```
ComfyUI/
├── user/default/workflows/
│   ├── dataset_gen_qie2511.json
│   ├── final_gen_zimage_base.json
│   └── prompts/
│       └── dataset_diversity_prompts.txt
├── scripts/
│   ├── caption_cleanup.py
│   └── caption_blocklist.txt
├── input/
│   └── reference/           # User drops reference photos here
└── output/
    └── dataset_{name}/      # Generated dataset output
```

---

## User-Facing Configuration

### Dataset Generation Workflow

| Widget | Default | Description |
|---|---|---|
| Reference image(s) | — | LoadImage: drop 1-5 reference photos |
| Trigger word | `ohwx person` | String node at top of workflow |
| Output folder | `output/dataset` | SaveImageKJ output_folder |
| Filename prefix | `img` | SaveImageKJ filename_prefix |
| Lightning toggle | OFF | Bypass switch on Lightning LoRA (30 vs 8 steps) |
| Seed | random | KSampler seed |

### Final Generation Workflow

| Widget | Default | Description |
|---|---|---|
| Prompt | — | Include trigger word in prompt |
| LoRA path | — | LoraLoaderModelOnly: trained identity LoRA |
| LoRA strength | 0.80 | Identity LoRA strength |
| Style LoRA | none | Optional second LoRA for style |
| Aspect ratio | 3:2 photo | EmptyZImageLatentImage |
| Face refine | ON | Bypass switch for SAM3 stage |
| Face restore | OFF | Bypass switch for CodeFormer |
| Upscale | OFF | Bypass switch for UltimateSDUpscale |
| Seed | random | KSampler seed |

---

## Model Requirements

### Dataset Generation

| Model | Path | Size | Status |
|---|---|---|---|
| qwen_image_edit_2511_bf16.safetensors | diffusion_models/ | 39 GB | Installed |
| qwen_2.5_vl_7b_fp8_scaled.safetensors | text_encoders/ | 8.8 GB | Installed |
| qwen_image_vae.safetensors | vae/ | 0.24 GB | Installed |
| Lightning 8-step LoRA (optional) | loras/ | 1.6 GB | In model registry |
| WD14 tagger ONNX | Auto-downloads | ~0.4 GB | Auto |
| JoyCaption LLM | models/LLM/ | ~4-17 GB | Auto-downloads |

### Final Generation

| Model | Path | Size | Status |
|---|---|---|---|
| z_image_bf16.safetensors | diffusion_models/ | 12.3 GB | Installed |
| qwen_3_4b.safetensors | text_encoders/ | 8 GB | Installed |
| z_image_ae.safetensors | vae/ | 0.34 GB | Installed |
| Identity LoRA | loras/ | varies | User-trained |
| SAM3 model | sam3/ | ~1 GB | Verify |
| codeformer.pth | facerestore_models/ | ~0.4 GB | Verify |
| 4x-UltraSharp.pth | upscale_models/ | ~0.07 GB | Verify |

---

## Implementation Validation Checklist

- [ ] SAM3Grounding with text "face" produces reliable masks on Z-Image outputs
- [ ] SaveImageKJ writes matching .txt files with caption STRING content
- [ ] JoyCaption "omit physical characteristics" mode works in installed version
- [ ] WD14 tagger produces useful tags on QIE2511 output images
- [ ] QIE2511 maintains identity across diverse prompt variations
- [ ] Z-Image Base + LoRA at 0.75-0.85 produces strong identity retention
- [ ] Face inpaint at denoise 0.40-0.45 improves faces without artifacts
- [ ] CodeFormer bypass switch (ComfyUI group mute/bypass) works cleanly
- [ ] UltimateSDUpscale bypass switch (ComfyUI group mute/bypass) works cleanly
- [ ] Caption cleanup script handles edge cases (empty captions, missing files)
- [ ] All 72 diversity prompts parse correctly in CR Prompt List
- [ ] Lightning LoRA bypass toggle correctly switches between 30 and 8 steps
