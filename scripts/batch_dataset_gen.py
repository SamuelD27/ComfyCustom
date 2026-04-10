#!/usr/bin/env python3
"""Batch dataset generation script for ComfyUI API.

Reads prompts from a file and queues each one through the ComfyUI HTTP API
to generate dataset images in batch. Uses only Python stdlib (no external
dependencies).

This script generates images WITHOUT JoyCaption recaptioning. Users either:
1. Use the full UI workflow for integrated captioning
2. Use this batch script for image generation, then run a separate captioning pass
"""

import argparse
import json
import logging
import random
import sys
import time
import urllib.error
import urllib.request

log = logging.getLogger(__name__)


def load_prompts(path: str) -> list[str]:
    """Read prompts file, return list of non-empty stripped lines.

    Each line is one prompt. Blank lines and lines starting with # are skipped.
    """
    with open(path, "r") as f:
        lines = []
        for line in f:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                lines.append(stripped)
    return lines


def check_comfyui(api_url: str) -> bool:
    """Check if ComfyUI is running at the URL.

    Sends GET /system_stats and returns True if a 200 response is received.
    """
    url = f"{api_url.rstrip('/')}/system_stats"
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status == 200
    except (urllib.error.URLError, OSError, TimeoutError):
        return False


def queue_prompt(
    api_url: str,
    prompt_text: str,
    trigger_word: str,
    reference_image: str = "reference.png",
    seed: int | None = None,
    output_folder: str = "output/dataset",
    steps: int = 30,
    cfg: float = 5.0,
) -> dict:
    """Queue a single generation via the ComfyUI API.

    Builds a ComfyUI API-format prompt dict and POSTs it to /prompt.

    Args:
        api_url: Base URL of the ComfyUI instance.
        prompt_text: The generation prompt (trigger word is prepended).
        trigger_word: Trigger word to prepend to the prompt.
        reference_image: Filename of the reference image in ComfyUI input folder.
        seed: Fixed seed, or None for random.
        output_folder: Output folder for generated images.
        steps: Number of sampling steps.
        cfg: Guidance scale.

    Returns:
        The JSON response from ComfyUI.

    Raises:
        urllib.error.URLError: If the request fails.
    """
    full_prompt = f"{trigger_word}, {prompt_text}" if trigger_word else prompt_text
    actual_seed = seed if seed is not None else random.randint(0, 2**32 - 1)

    prompt_dict = {
        "1": {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": "qwen_image_edit_2511_bf16.safetensors",
                "weight_dtype": "default",
            },
        },
        "3": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": "qwen_2.5_vl_7b_fp8_scaled.safetensors",
                "type": "qwen2_5_vl",
            },
        },
        "4": {
            "class_type": "VAELoader",
            "inputs": {
                "vae_name": "qwen_image_vae.safetensors",
            },
        },
        "5": {
            "class_type": "LoadImage",
            "inputs": {
                "image": reference_image,
            },
        },
        "10": {
            "class_type": "TextEncodeQwenImageEditPlus_lrzjason",
            "inputs": {
                "clip": ["3", 0],
                "vae": ["4", 0],
                "prompt": full_prompt,
                "image1": ["5", 0],
                "enable_resize": True,
                "resolution": 1024,
                "instruction": (
                    "Describe the key features of the input image "
                    "and generate an image based on the prompt."
                ),
            },
        },
        "11": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": "",
                "clip": ["3", 0],
            },
        },
        "15": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["1", 0],
                "positive": ["10", 0],
                "negative": ["11", 0],
                "latent_image": ["10", 6],
                "seed": actual_seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": "euler",
                "scheduler": "beta57",
                "denoise": 1.0,
            },
        },
        "16": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["15", 0],
                "vae": ["4", 0],
            },
        },
        "30": {
            "class_type": "SaveImageKJ",
            "inputs": {
                "images": ["16", 0],
                "filename_prefix": "img",
                "output_folder": output_folder,
                "caption_file_extension": ".txt",
            },
        },
    }

    url = f"{api_url.rstrip('/')}/prompt"
    payload = json.dumps({"prompt": prompt_dict}).encode("utf-8")
    req = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main():
    parser = argparse.ArgumentParser(
        description="Batch dataset generation via ComfyUI API. "
        "Reads prompts from a file and queues each one for image generation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  %(prog)s --prompts prompts.txt\n"
            "  %(prog)s --prompts prompts.txt --start 10 --count 5\n"
            "  %(prog)s --prompts prompts.txt --seed 42 --steps 20 --cfg 3.5\n"
        ),
    )
    parser.add_argument(
        "--api-url",
        default="http://127.0.0.1:8188",
        help="ComfyUI API base URL (default: http://127.0.0.1:8188)",
    )
    parser.add_argument(
        "--prompts",
        default="user/default/workflows/prompts/dataset_diversity_prompts.txt",
        help="Path to prompts file (default: user/default/workflows/prompts/dataset_diversity_prompts.txt)",
    )
    parser.add_argument(
        "--reference",
        default="reference.png",
        help="Reference image filename in ComfyUI input folder (default: reference.png)",
    )
    parser.add_argument(
        "--output-folder",
        default="output/dataset",
        help="Output folder for generated images (default: output/dataset)",
    )
    parser.add_argument(
        "--trigger",
        default="ohwx person",
        help="Trigger word to prepend to prompts (default: ohwx person)",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start from prompt index N (default: 0)",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=None,
        help="Generate N images (default: all remaining prompts)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help="Seconds between queue submissions (default: 2.0)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Fixed seed for all images (default: random per image)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=30,
        help="Sampling steps (default: 30)",
    )
    parser.add_argument(
        "--cfg",
        type=float,
        default=5.0,
        help="Guidance scale (default: 5.0)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
    )

    # 1. Check ComfyUI is running
    log.info("Checking ComfyUI at %s ...", args.api_url)
    if not check_comfyui(args.api_url):
        log.error("ERROR: ComfyUI is not reachable at %s", args.api_url)
        log.error("Make sure ComfyUI is running (./start.sh) and try again.")
        sys.exit(1)
    log.info("ComfyUI is running.")

    # 2. Load prompts from file
    try:
        prompts = load_prompts(args.prompts)
    except FileNotFoundError:
        log.error("ERROR: Prompts file not found: %s", args.prompts)
        sys.exit(1)

    if not prompts:
        log.error("ERROR: No prompts found in %s", args.prompts)
        sys.exit(1)

    log.info("Loaded %d prompts from %s", len(prompts), args.prompts)

    # 3. Select range
    start = args.start
    end = start + args.count if args.count is not None else len(prompts)
    selected = prompts[start:end]

    if not selected:
        log.error(
            "ERROR: No prompts in range [%d:%d] (file has %d prompts)",
            start,
            end,
            len(prompts),
        )
        sys.exit(1)

    log.info(
        "Generating %d images (prompts %d-%d)",
        len(selected),
        start,
        start + len(selected) - 1,
    )
    log.info("  Trigger: %s", args.trigger)
    log.info("  Reference: %s", args.reference)
    log.info("  Output: %s", args.output_folder)
    log.info("  Steps: %d, CFG: %.1f", args.steps, args.cfg)
    log.info("  Seed: %s", args.seed if args.seed is not None else "random")
    log.info("")

    # 4. Queue each prompt
    queued = 0
    failed = 0

    for i, prompt_text in enumerate(selected):
        idx = start + i
        try:
            result = queue_prompt(
                api_url=args.api_url,
                prompt_text=prompt_text,
                trigger_word=args.trigger,
                reference_image=args.reference,
                seed=args.seed,
                output_folder=args.output_folder,
                steps=args.steps,
                cfg=args.cfg,
            )
            prompt_id = result.get("prompt_id", "unknown")
            log.info(
                "[%d/%d] Queued prompt %d: %.60s... (id: %s)",
                i + 1,
                len(selected),
                idx,
                prompt_text,
                prompt_id,
            )
            queued += 1
        except (urllib.error.URLError, OSError, json.JSONDecodeError) as e:
            log.error(
                "[%d/%d] FAILED prompt %d: %.60s... (%s)",
                i + 1,
                len(selected),
                idx,
                prompt_text,
                e,
            )
            failed += 1

        # Sleep between submissions (skip after last one)
        if i < len(selected) - 1 and args.delay > 0:
            time.sleep(args.delay)

    # 5. Print summary
    log.info("")
    log.info("Done. Queued: %d, Failed: %d, Total: %d", queued, failed, len(selected))

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
