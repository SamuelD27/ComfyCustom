# ComfyUI Backend Instance

**Version:** 0.11.1
**Role:** Backend image generation server for other projects
**Hardware:** ASUS GX10 / NVIDIA GB10 (128 GB VRAM)

---

## Quick Reference

```bash
# Start ComfyUI (listens on all interfaces, port 8188)
./start.sh

# Start with specific port
./start.sh --port 8190

# Stop
pkill -f "python.*main.py"

# Check if running
curl -s http://localhost:8188/system_stats | python3 -m json.tool
```

---

## Architecture

This is a **stock ComfyUI install from source** (not pip-installed), used purely as a backend API server. The frontend is served but the primary consumers are external projects hitting the API.

### Key Paths
- **Entry point:** `main.py`
- **Server:** `server.py` (aiohttp-based)
- **Node definitions:** `nodes.py`, `comfy_extras/`
- **Custom nodes:** `custom_nodes/`
- **Models:** `models/` (gitignored)
- **Python env:** `.venv/` (Python 3.12)

### Core Modules
| Directory | Purpose |
|---|---|
| `comfy/` | Core inference engine, model management, samplers |
| `comfy_api/` | API node definitions and feature flags |
| `comfy_api_nodes/` | Extended API-accessible nodes |
| `comfy_execution/` | Execution graph, caching, scheduling |
| `comfy_extras/` | Built-in extra nodes (ControlNet, IP-Adapter, etc.) |
| `api_server/` | REST API endpoints |
| `app/` | Application setup, logger, assets |
| `middleware/` | HTTP middleware |

---

## Installed Custom Nodes

| Node | Purpose |
|---|---|
| `ComfyUI-AutomaticCFG` | Automatic CFG scaling |
| `ComfyUI-Detail-Daemon` | Detail enhancement |
| `ComfyUI-Impact-Pack` | Detection, segmentation, detailers |
| `ComfyUI_InfiniteYou` | Identity-preserving generation |
| `ComfyUI-ppm` | Perturbed-attention guidance |
| `ComfyUI-PuLID-Flux` | PuLID face identity for Flux |
| `ComfyUI_UltimateSDUpscale` | Tiled upscaling |
| `facerestore_advanced` | Advanced face restoration |
| `facerestore_cf` | CodeFormer face restoration |
| `sd-perturbed-attention` | Perturbed attention guidance |
| `x-flux-comfyui` | X-Flux integration |
| `websocket_image_save.py` | WebSocket image output |

---

## Installed Models

### Flux Pipeline (Primary)
- **Checkpoint:** `flux1-dev.safetensors`
- **Diffusion model:** `flux1-dev.safetensors`
- **Text encoders:** `clip_l.safetensors`, `t5xxl_fp8_e4m3fn.safetensors`
- **VAE:** `flux1-ae.safetensors`

### Face / Identity
- **PuLID:** `pulid_flux_v0.9.1.safetensors`
- **InfiniteYou:** `infu_flux_v1.0`, `aes_stage2`, `sim_stage1`
- **InsightFace:** `models/insightface/models/`
- **Face detection:** `detection_Resnet50_Final.pth`, `parsing_parsenet.pth`
- **Face restore:** `codeformer.pth`, `GFPGANv1.4.pth`, `GPEN-BFR-512.pth`

### Upscale
- `4x-UltraSharp.pth`
- `RealESRGAN_x4plus.pth`

### Segmentation
- **SAM:** `sam_vit_b_01ec64.pth`

### XLabs (empty dirs scaffolded)
- `controlnets/`, `flux/`, `ipadapters/`, `loras/`

---

## Development Notes

### Running as Backend
When using ComfyUI as a backend for other projects, start with:
```bash
./start.sh --enable-cors-header
```
This enables CORS and listens on `0.0.0.0:8188` so other services can reach it.

### VRAM (128 GB)
With 128 GB VRAM on the GB10, use `--highvram` to keep models in VRAM for faster inference. This is the default in `start.sh`.

### Adding Custom Nodes
```bash
cd custom_nodes
git clone <repo-url>
# Restart ComfyUI to load
```

### Adding Models
Drop model files into the appropriate `models/<type>/` subdirectory. ComfyUI picks them up on next API call or restart.

### API Usage
- **Queue prompt:** `POST /prompt` with workflow JSON
- **Get history:** `GET /history`
- **Get system stats:** `GET /system_stats`
- **Upload image:** `POST /upload/image`
- **View image:** `GET /view?filename=...`
- **WebSocket:** `ws://host:8188/ws` for real-time progress

### Testing
```bash
# Unit tests
.venv/bin/python -m pytest tests-unit/
```

### Linting
Configured via `pyproject.toml` with ruff. Key ignores: E501 (line length), E722, E402.

### Model Manager (Temporary Use System)

The startup script includes an interactive model manager that downloads models on demand and cleans them up on exit.

**How it works:**
1. `./start.sh` shows disk space and launches the model manager
2. Select which pipeline(s) to load (Flux 2 Dev, Flux 2 Klein 9B, Qwen Image Edit 2511)
3. Select optional LoRAs for each pipeline
4. Models download with progress bars (HF via `hf` CLI, CivitAI via aria2c)
5. On Ctrl+C, all temporary models + HF cache blobs are deleted to free disk space

**Key files:**
- `scripts/model_registry.json` -- model definitions, download URLs, sizes
- `scripts/model_manager.py` -- interactive UI, downloads, cleanup
- `models/.tmp_models_manifest.json` -- runtime tracking (auto-deleted on cleanup)

**AI Toolkit LoRAs** in `~/ai-toolkit/output/` are always symlinked and never cleaned up.

**Adding new models:** Edit `scripts/model_registry.json` to add pipelines, models, or LoRAs.

**Logs:** Technical logs go to `comfyui.log`, terminal shows colorized filtered output.
