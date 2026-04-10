#!/bin/bash
# Download upscale models for flux2_klein_9b_upscale workflow
UPSCALE_DIR="ComfyUI/models/upscale_models"
mkdir -p "$UPSCALE_DIR"

echo "Downloading 4xNomosUniDAT_otf (primary - best for portraits)..."
hf download Phips/4xNomosUniDAT_otf 4xNomosUniDAT_otf.safetensors \
  --local-dir "$UPSCALE_DIR" --max-workers 32

echo "Downloading 4xRealWebPhoto_RGT (backup - best for AI-generated images)..."
wget -c "https://github.com/Phhofm/models/releases/download/4xRealWebPhoto_RGT/4xRealWebPhoto_RGT.pth" \
  -O "$UPSCALE_DIR/4xRealWebPhoto_RGT.pth"

echo "Downloading 4x-UltraSharp (fallback)..."
wget -c "https://huggingface.co/datasets/BlodyTraveler/4x-UltraSharp/resolve/main/4x-UltraSharp.pth" \
  -O "$UPSCALE_DIR/4x-UltraSharp.pth"

echo "All upscale models downloaded to $UPSCALE_DIR"
