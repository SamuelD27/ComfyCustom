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
            "mode": 4,  # BYPASSED — passes MODEL through when disabled
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
                "qwen_image",
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
            "type": "CR Text",
            "title": "Trigger Word",
            "pos": [550, 550],
            "size": [315, 82],
            "inputs": [],
            "outputs": [("*", "*"), ("STRING", "STRING")],
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
                "",         # prompt (placeholder — overridden by link)
                True,       # enable_resize
                True,       # enable_vl_resize
                False,      # skip_first_image_resize
                "bicubic",  # upscale_method
                "disabled", # crop
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

        # -- Crop right half (QIE2511 outputs reference|generated side-by-side) --
        {
            "id": 60,
            "type": "GetImageSize",
            "title": "Get Output Size",
            "pos": [2150, 100],
            "size": [210, 46],
            "inputs": [
                ("image", "IMAGE"),
            ],
            "outputs": [
                ("width", "INT"),
                ("height", "INT"),
                ("batch_size", "INT"),
            ],
            "widgets_values": [],
        },
        {
            "id": 61,
            "type": "CR Integer Multiple",
            "title": "Width / 2",
            "pos": [2150, 220],
            "size": [250, 82],
            "inputs": [
                ("integer", "INT"),
            ],
            "outputs": [
                ("INT", "INT"),
                ("show_help", "STRING"),
            ],
            "widgets_values": [0, 0.5],
        },
        {
            "id": 62,
            "type": "ImageCrop",
            "title": "Crop Right Half (generated image)",
            "pos": [2150, 380],
            "size": [315, 130],
            "inputs": [
                ("image", "IMAGE"),
                ("width", "INT"),
                ("height", "INT"),
                ("x", "INT"),
                ("y", "INT"),
            ],
            "outputs": [
                ("IMAGE", "IMAGE"),
            ],
            "widgets_values": [512, 512, 0, 0],
        },

        # -- Captioning (x=2500) --
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
                "Full Precision (bf16)",     # quantization (no bitsandbytes — GB10 incompatible)
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

        # Crop right half of QIE2511 output (left=reference, right=generated)
        (16, 0, 60, 0, "IMAGE"),    # VAEDecode -> GetImageSize
        (60, 0, 61, 0, "INT"),      # width -> CR Integer Multiple (÷2)
        (16, 0, 62, 0, "IMAGE"),    # VAEDecode -> ImageCrop image
        (61, 0, 62, 1, "INT"),      # half_width -> ImageCrop width
        (60, 1, 62, 2, "INT"),      # height -> ImageCrop height
        (61, 0, 62, 3, "INT"),      # half_width -> ImageCrop x
        # y=0 stays as widget default

        # Cropped image to JoyCaption
        (62, 0, 21, 0, "IMAGE"),    # Cropped IMAGE -> JC_adv image

        # Extra options to JoyCaption
        (20, 0, 21, 1, "JOYCAPTION_EXTRA_OPTIONS"),  # JC_ExtraOptions -> JC_adv

        # Trigger word to JoinStrings
        (8, 0, 25, 0, "STRING"),    # PrimitiveNode -> JoinStrings string1

        # Caption to JoinStrings
        (21, 1, 25, 1, "STRING"),   # JC_adv STRING output (slot 1) -> JoinStrings string2

        # Cropped image to SaveImageKJ
        (62, 0, 30, 0, "IMAGE"),    # Cropped IMAGE -> SaveImageKJ images

        # Joined caption to SaveImageKJ
        (25, 0, 30, 1, "STRING"),   # JoinStrings -> SaveImageKJ caption

        # Cropped image to preview
        (62, 0, 31, 0, "IMAGE"),    # Cropped IMAGE -> PreviewImage
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


def build_final_gen():
    """Build the Z-Image Base i2i final generation workflow.

    Image-to-image pipeline with multiple reference inputs:
      Reference: 3 LoadImage slots (primary sets composition, 2 optional)
      Stage 1: i2i generation (Z-Image + identity LoRA, denoise 0.65)
      Stage 2: Face refinement (SAM3 + inpaint crop/stitch)
      Stage 3: Face restoration (CodeFormer, optional/bypassed)
      Stage 4: Upscale (4x, optional/bypassed)
    """

    nodes_def = [
        # =================================================================
        # STAGE 1: Base Generation
        # =================================================================

        # -- Model Loaders --
        {
            "id": 1,
            "type": "UNETLoader",
            "title": "Load Z-Image Diffusion Model",
            "pos": [100, 100],
            "size": [315, 82],
            "inputs": [],
            "outputs": [("MODEL", "MODEL")],
            "widgets_values": [
                "z_image_bf16.safetensors",
                "default",
            ],
        },
        {
            "id": 2,
            "type": "ModelSamplingAuraFlow",
            "title": "Model Sampling (shift=3.0)",
            "pos": [100, 250],
            "size": [315, 82],
            "inputs": [
                ("model", "MODEL"),
            ],
            "outputs": [("MODEL", "MODEL")],
            "widgets_values": [3.0],
        },
        {
            "id": 3,
            "type": "LoraLoaderModelOnly",
            "title": "Identity LoRA",
            "pos": [100, 400],
            "size": [315, 82],
            "inputs": [
                ("model", "MODEL"),
            ],
            "outputs": [("MODEL", "MODEL")],
            "widgets_values": [
                "identity_lora.safetensors",
                0.80,
            ],
        },
        {
            "id": 4,
            "type": "LoraLoaderModelOnly",
            "title": "Style LoRA (optional)",
            "pos": [100, 550],
            "size": [315, 82],
            "mode": 2,  # MUTED
            "inputs": [
                ("model", "MODEL"),
            ],
            "outputs": [("MODEL", "MODEL")],
            "widgets_values": [
                "style_lora.safetensors",
                1.0,
            ],
        },
        {
            "id": 5,
            "type": "CLIPLoader",
            "title": "Load Qwen3-4B Text Encoder",
            "pos": [100, 700],
            "size": [315, 82],
            "inputs": [],
            "outputs": [("CLIP", "CLIP")],
            "widgets_values": [
                "qwen_3_4b.safetensors",
                "lumina2",
            ],
        },
        {
            "id": 6,
            "type": "VAELoader",
            "title": "Load Z-Image VAE",
            "pos": [100, 850],
            "size": [315, 82],
            "inputs": [],
            "outputs": [("VAE", "VAE")],
            "widgets_values": [
                "z_image_ae.safetensors",
            ],
        },

        # -- Reference Images (i2i inputs) --
        {
            "id": 50,
            "type": "LoadImage",
            "title": "Reference 1 (Primary — sets composition)",
            "pos": [550, 100],
            "size": [315, 314],
            "inputs": [],
            "outputs": [
                ("IMAGE", "IMAGE"),
                ("MASK", "MASK"),
            ],
            "widgets_values": ["example.png", "image"],
        },
        {
            "id": 51,
            "type": "LoadImage",
            "title": "Reference 2 (optional)",
            "pos": [550, 470],
            "size": [315, 314],
            "mode": 4,  # BYPASSED by default
            "inputs": [],
            "outputs": [
                ("IMAGE", "IMAGE"),
                ("MASK", "MASK"),
            ],
            "widgets_values": ["example.png", "image"],
        },
        {
            "id": 52,
            "type": "LoadImage",
            "title": "Reference 3 (optional)",
            "pos": [550, 840],
            "size": [315, 314],
            "mode": 4,  # BYPASSED by default
            "inputs": [],
            "outputs": [
                ("IMAGE", "IMAGE"),
                ("MASK", "MASK"),
            ],
            "widgets_values": ["example.png", "image"],
        },

        # -- Scale + Encode primary reference --
        {
            "id": 53,
            "type": "ImageScaleToTotalPixels",
            "title": "Scale to Target Resolution",
            "pos": [950, 100],
            "size": [315, 82],
            "inputs": [
                ("image", "IMAGE"),
            ],
            "outputs": [
                ("IMAGE", "IMAGE"),
            ],
            "widgets_values": [
                "bilinear",     # upscale_method
                1.0,            # megapixels (1MP = ~1024x1024)
            ],
        },
        {
            "id": 54,
            "type": "VAEEncode",
            "title": "Encode Reference to Latent",
            "pos": [950, 270],
            "size": [210, 46],
            "inputs": [
                ("pixels", "IMAGE"),
                ("vae", "VAE"),
            ],
            "outputs": [
                ("LATENT", "LATENT"),
            ],
            "widgets_values": [],
        },

        # -- Prompts --
        {
            "id": 10,
            "type": "CLIPTextEncode",
            "title": "Positive Prompt",
            "pos": [950, 400],
            "size": [400, 200],
            "inputs": [
                ("clip", "CLIP"),
            ],
            "outputs": [
                ("CONDITIONING", "CONDITIONING"),
            ],
            "widgets_values": [
                "ohwx person, a photorealistic portrait photograph",
            ],
        },
        {
            "id": 11,
            "type": "CLIPTextEncode",
            "title": "Negative Prompt",
            "pos": [950, 650],
            "size": [400, 200],
            "inputs": [
                ("clip", "CLIP"),
            ],
            "outputs": [
                ("CONDITIONING", "CONDITIONING"),
            ],
            "widgets_values": [
                "blurry, deformed, distorted face, extra fingers, "
                "bad anatomy, watermark, text, low quality",
            ],
        },

        # -- Base Sampler (i2i with denoise < 1.0) --
        {
            "id": 15,
            "type": "KSampler",
            "title": "Base Sampler (i2i)",
            "pos": [1450, 100],
            "size": [315, 262],
            "inputs": [
                ("model", "MODEL"),
                ("positive", "CONDITIONING"),
                ("negative", "CONDITIONING"),
                ("latent_image", "LATENT"),
            ],
            "outputs": [
                ("LATENT", "LATENT"),
            ],
            "widgets_values": [
                0,                      # seed
                "randomize",            # seed control
                25,                     # steps
                4.0,                    # cfg
                "euler",                # sampler_name
                "linear_quadratic",     # scheduler
                0.65,                   # denoise (i2i: preserve reference structure)
            ],
        },

        # -- Base Decode --
        {
            "id": 16,
            "type": "VAEDecode",
            "title": "VAE Decode (Base)",
            "pos": [1450, 450],
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

        # =================================================================
        # STAGE 2: Face Refinement (SAM3 + Inpaint)
        # =================================================================

        # -- SAM3 --
        {
            "id": 20,
            "type": "LoadSAM3Model",
            "title": "Load SAM3 Model",
            "pos": [1500, 100],
            "size": [315, 82],
            "inputs": [],
            "outputs": [
                ("SAM3_MODEL", "SAM3_MODEL"),
            ],
            "widgets_values": [
                "sam3.safetensors",
            ],
        },
        {
            "id": 21,
            "type": "SAM3Grounding",
            "title": "SAM3 Detect Face",
            "pos": [1500, 250],
            "size": [315, 150],
            "inputs": [
                ("sam3_model", "SAM3_MODEL"),
                ("image", "IMAGE"),
            ],
            "outputs": [
                ("masks", "MASK"),
                ("visualization", "IMAGE"),
                ("boxes", "STRING"),
                ("scores", "STRING"),
            ],
            "widgets_values": [
                0.33,       # confidence_threshold
                "face",     # text_prompt
                1,          # max_detections
            ],
        },

        # -- Inpaint Crop --
        {
            "id": 22,
            "type": "InpaintCropImproved",
            "title": "Inpaint Crop (Face)",
            "pos": [1500, 470],
            "size": [315, 500],
            "inputs": [
                ("image", "IMAGE"),
                ("mask", "MASK"),
                ("optional_context_mask", "MASK"),
            ],
            "outputs": [
                ("stitcher", "STITCHER"),
                ("cropped_image", "IMAGE"),
                ("cropped_mask", "MASK"),
            ],
            "widgets_values": [
                "bilinear",     # downscale_algorithm
                "bicubic",      # upscale_algorithm
                False,          # preresize
                "ensure minimum resolution",  # preresize_mode
                1024,           # preresize_min_width
                1024,           # preresize_min_height
                16384,          # preresize_max_width
                16384,          # preresize_max_height
                True,           # mask_fill_holes
                0,              # mask_expand_pixels
                False,          # mask_invert
                32,             # mask_blend_pixels
                0.1,            # mask_hipass_filter
                False,          # extend_for_outpainting
                1.0,            # extend_up_factor
                1.0,            # extend_down_factor
                1.0,            # extend_left_factor
                1.0,            # extend_right_factor
                2.0,            # context_from_mask_extend_factor
                True,           # output_resize_to_target_size
                512,            # output_target_width
                512,            # output_target_height
                "32",           # output_padding
                "gpu (much faster)",  # device_mode
            ],
        },

        # -- Inpaint Conditioning --
        {
            "id": 24,
            "type": "InpaintModelConditioning",
            "title": "Inpaint Conditioning",
            "pos": [1900, 100],
            "size": [315, 200],
            "inputs": [
                ("positive", "CONDITIONING"),
                ("negative", "CONDITIONING"),
                ("vae", "VAE"),
                ("pixels", "IMAGE"),
                ("mask", "MASK"),
            ],
            "outputs": [
                ("positive", "CONDITIONING"),
                ("negative", "CONDITIONING"),
                ("latent", "LATENT"),
            ],
            "widgets_values": [True],  # noise_mask
        },

        # -- Identity LoRA (higher strength for face) --
        {
            "id": 45,
            "type": "LoraLoaderModelOnly",
            "title": "Identity LoRA (Face, 0.95)",
            "pos": [1900, 370],
            "size": [315, 82],
            "inputs": [
                ("model", "MODEL"),
            ],
            "outputs": [("MODEL", "MODEL")],
            "widgets_values": [
                "identity_lora.safetensors",
                0.95,
            ],
        },

        # -- Face Sampler --
        {
            "id": 25,
            "type": "KSampler",
            "title": "Face Sampler",
            "pos": [1900, 520],
            "size": [315, 262],
            "inputs": [
                ("model", "MODEL"),
                ("positive", "CONDITIONING"),
                ("negative", "CONDITIONING"),
                ("latent_image", "LATENT"),
            ],
            "outputs": [
                ("LATENT", "LATENT"),
            ],
            "widgets_values": [
                0,                      # seed
                "randomize",            # seed control
                18,                     # steps
                3.0,                    # cfg
                "euler",                # sampler_name
                "linear_quadratic",     # scheduler
                0.42,                   # denoise
            ],
        },

        # -- Decode + Stitch --
        {
            "id": 26,
            "type": "VAEDecode",
            "title": "VAE Decode (Face)",
            "pos": [2300, 100],
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
        {
            "id": 27,
            "type": "InpaintStitchImproved",
            "title": "Inpaint Stitch (Face)",
            "pos": [2300, 220],
            "size": [315, 82],
            "inputs": [
                ("stitcher", "STITCHER"),
                ("inpainted_image", "IMAGE"),
            ],
            "outputs": [
                ("image", "IMAGE"),
            ],
            "widgets_values": [],
        },

        # =================================================================
        # STAGE 3: Face Restore (optional, muted)
        # =================================================================
        {
            "id": 30,
            "type": "FaceRestoreModelLoader",
            "title": "Load CodeFormer",
            "pos": [2700, 100],
            "size": [315, 82],
            "mode": 2,  # MUTED
            "inputs": [],
            "outputs": [
                ("FACERESTORE_MODEL", "FACERESTORE_MODEL"),
            ],
            "widgets_values": [
                "codeformer.pth",
            ],
        },
        {
            "id": 31,
            "type": "FaceRestoreCFWithModel",
            "title": "CodeFormer Restore",
            "pos": [2700, 250],
            "size": [315, 150],
            "mode": 4,  # BYPASSED (passes IMAGE through)
            "inputs": [
                ("facerestore_model", "FACERESTORE_MODEL"),
                ("image", "IMAGE"),
            ],
            "outputs": [
                ("IMAGE", "IMAGE"),
            ],
            "widgets_values": [
                "retinaface_resnet50",  # facedetection
                0.7,                    # codeformer_fidelity
            ],
        },

        # =================================================================
        # STAGE 4: Upscale (optional, muted)
        # =================================================================
        {
            "id": 35,
            "type": "UpscaleModelLoader",
            "title": "Load 4x Upscale Model",
            "pos": [3100, 100],
            "size": [315, 82],
            "mode": 2,  # MUTED
            "inputs": [],
            "outputs": [
                ("UPSCALE_MODEL", "UPSCALE_MODEL"),
            ],
            "widgets_values": [
                "4xNomosUniDAT_otf.pth",
            ],
        },
        {
            "id": 36,
            "type": "ImageUpscaleWithModel",
            "title": "4x Upscale",
            "pos": [3100, 250],
            "size": [315, 82],
            "mode": 4,  # BYPASSED (passes IMAGE through)
            "inputs": [
                ("upscale_model", "UPSCALE_MODEL"),
                ("image", "IMAGE"),
            ],
            "outputs": [
                ("IMAGE", "IMAGE"),
            ],
            "widgets_values": [],
        },

        # =================================================================
        # OUTPUT
        # =================================================================
        {
            "id": 40,
            "type": "SaveImage",
            "title": "Save Image",
            "pos": [3500, 100],
            "size": [315, 270],
            "inputs": [
                ("images", "IMAGE"),
            ],
            "outputs": [],
            "widgets_values": [
                "zimage_final",     # filename_prefix
            ],
        },
        {
            "id": 41,
            "type": "PreviewImage",
            "title": "Preview",
            "pos": [3500, 430],
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
        # --- Stage 1: Model chain ---
        (1, 0, 2, 0, "MODEL"),      # UNETLoader -> ModelSampling model
        (2, 0, 3, 0, "MODEL"),      # ModelSampling -> Identity LoRA model
        (3, 0, 4, 0, "MODEL"),      # Identity LoRA -> Style LoRA model
        (4, 0, 15, 0, "MODEL"),     # Style LoRA -> Base KSampler model

        # CLIP to both text encoders
        (5, 0, 10, 0, "CLIP"),      # CLIPLoader -> Positive prompt clip
        (5, 0, 11, 0, "CLIP"),      # CLIPLoader -> Negative prompt clip

        # --- Reference image pipeline ---
        # Primary reference -> scale -> encode -> latent for i2i
        (50, 0, 53, 0, "IMAGE"),    # Reference 1 -> ImageScale
        (53, 0, 54, 0, "IMAGE"),    # Scaled image -> VAEEncode pixels
        (6, 0, 54, 1, "VAE"),       # VAE -> VAEEncode vae

        # VAE to base decode
        (6, 0, 16, 1, "VAE"),       # VAELoader -> Base VAEDecode vae

        # Conditioning to base sampler
        (10, 0, 15, 1, "CONDITIONING"),  # Positive -> Base KSampler positive
        (11, 0, 15, 2, "CONDITIONING"),  # Negative -> Base KSampler negative

        # Encoded reference latent to base sampler (i2i)
        (54, 0, 15, 3, "LATENT"),   # VAEEncode latent -> Base KSampler latent_image

        # Base sampler to decode
        (15, 0, 16, 0, "LATENT"),   # Base KSampler -> Base VAEDecode samples

        # --- Stage 2: Face Refinement ---
        # SAM3 face detection
        (16, 0, 21, 1, "IMAGE"),    # Base decoded image -> SAM3Grounding image
        (20, 0, 21, 0, "SAM3_MODEL"),  # LoadSAM3Model -> SAM3Grounding model

        # Inpaint crop
        (16, 0, 22, 0, "IMAGE"),    # Base decoded image -> InpaintCrop image
        (21, 0, 22, 1, "MASK"),     # SAM3 masks -> InpaintCrop mask

        # Inpaint conditioning
        (10, 0, 24, 0, "CONDITIONING"),  # Positive -> InpaintCond positive
        (11, 0, 24, 1, "CONDITIONING"),  # Negative -> InpaintCond negative
        (6, 0, 24, 2, "VAE"),            # VAE -> InpaintCond vae
        (22, 1, 24, 3, "IMAGE"),         # Cropped image -> InpaintCond pixels
        (22, 2, 24, 4, "MASK"),          # Cropped mask -> InpaintCond mask

        # Identity LoRA at higher strength for face sampler
        (4, 0, 45, 0, "MODEL"),     # Style LoRA out -> Face Identity LoRA model

        # Face sampler
        (45, 0, 25, 0, "MODEL"),         # Face Identity LoRA -> Face KSampler model
        (24, 0, 25, 1, "CONDITIONING"),  # InpaintCond positive -> Face KSampler
        (24, 1, 25, 2, "CONDITIONING"),  # InpaintCond negative -> Face KSampler
        (24, 2, 25, 3, "LATENT"),        # InpaintCond latent -> Face KSampler

        # Decode refined face
        (25, 0, 26, 0, "LATENT"),   # Face KSampler -> Face VAEDecode samples
        (6, 0, 26, 1, "VAE"),       # VAE -> Face VAEDecode vae

        # Stitch face back
        (22, 0, 27, 0, "STITCHER"),  # InpaintCrop stitcher -> InpaintStitch
        (26, 0, 27, 1, "IMAGE"),     # Decoded face -> InpaintStitch inpainted_image

        # --- Stage 3: Face Restore (muted) ---
        (30, 0, 31, 0, "FACERESTORE_MODEL"),  # FaceRestoreModelLoader -> FaceRestoreCF
        (27, 0, 31, 1, "IMAGE"),              # Stitched image -> FaceRestoreCF image

        # --- Stage 4: Upscale (muted) ---
        (35, 0, 36, 0, "UPSCALE_MODEL"),  # UpscaleModelLoader -> ImageUpscale
        (31, 0, 36, 1, "IMAGE"),           # Face restored image -> ImageUpscale

        # --- Output ---
        (36, 0, 40, 0, "IMAGE"),    # Upscaled image -> SaveImage
        (36, 0, 41, 0, "IMAGE"),    # Upscaled image -> PreviewImage
    ]

    # ---------------------------------------------------------------
    # Groups
    # ---------------------------------------------------------------
    groups_def = [
        {
            "title": "Reference Images",
            "bounding": [510, 50, 400, 1150],
            "color": "#533",
        },
        {
            "title": "Stage 1: i2i Generation",
            "bounding": [910, 50, 950, 850],
            "color": "#363",
        },
        {
            "title": "Stage 2: Face Refinement (SAM3)",
            "bounding": [1460, 50, 1100, 780],
            "color": "#336",
        },
        {
            "title": "Stage 3: Face Restore (optional)",
            "bounding": [2660, 50, 400, 380],
            "color": "#549",
        },
        {
            "title": "Stage 4: Upscale (optional)",
            "bounding": [3060, 50, 400, 310],
            "color": "#549",
        },
        {
            "title": "Output",
            "bounding": [3460, 50, 580, 930],
            "color": "#444",
        },
    ]

    return make_workflow(nodes_def, links_def, groups_def)


def main():
    # Dataset generation workflow
    wf_dataset = build_dataset_gen()
    out_dataset = PROJECT_ROOT / "user/default/workflows/qwen-image-edit/advanced/dataset_gen_qie2511.json"
    save_workflow(wf_dataset, out_dataset)

    print(f"\n--- Dataset Gen Workflow ---")
    print(f"Nodes: {len(wf_dataset['nodes'])}")
    print(f"Links: {len(wf_dataset['links'])}")
    for n in wf_dataset["nodes"]:
        mode_label = {2: " (MUTED)", 4: " (BYPASSED)"}.get(n.get("mode", 0), "")
        print(f"  {n['id']}: {n['type']}{mode_label} - {n.get('title', '')}")

    # Final generation workflow
    wf_final = build_final_gen()
    out_final = PROJECT_ROOT / "user/default/workflows/z-image-base/advanced/final_gen_zimage_base.json"
    save_workflow(wf_final, out_final)

    print(f"\n--- Final Gen Workflow ---")
    print(f"Nodes: {len(wf_final['nodes'])}")
    print(f"Links: {len(wf_final['links'])}")
    for n in wf_final["nodes"]:
        mode_label = {2: " (MUTED)", 4: " (BYPASSED)"}.get(n.get("mode", 0), "")
        print(f"  {n['id']}: {n['type']}{mode_label} - {n.get('title', '')}")


if __name__ == "__main__":
    main()
