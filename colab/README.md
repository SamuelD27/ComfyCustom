# ComfyUI on Colab — Personalised Launcher

One-click launch of this repo's ComfyUI setup on Google Colab, with registry-driven
pipeline selection, on-demand weight downloads, and a public `cloudflared` URL.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/samsam27/ComfyUI/blob/colab/colab/ComfyUI_Colab.ipynb)

The launcher lives on the long-lived `colab` branch (parallel to `master`) so the Colab-specific
code never merges into the main repo. Push this worktree to `origin/colab` on your fork before
the Open-in-Colab badge will resolve. Update the badge link if you fork this repo: replace
`samsam27/ComfyUI` with `<your-gh-user>/ComfyUI`.

## Prerequisites

- **Colab Pro+** (A100/G4 access and background execution).
- **Drive permission** — prompted on first run; used only for a ~1 KB `prefs.json`.
- **Secrets** added to Colab's key-icon sidebar (optional but recommended):
  - `HF_TOKEN` — HuggingFace access token
  - `CIVITAI_TOKEN` — CivitAI API key (for LoRA downloads)

If a secret is missing, the notebook falls back to a hidden `getpass` prompt in the cell.

## How it works

The notebook has eight numbered sections, run top-to-bottom:

1. Mount Drive (soft-fail if denied).
2. GPU detect + conditional torch reinstall (auto-restart prompt if cu128 was needed).
3. Load `HF_TOKEN` / `CIVITAI_TOKEN` into env vars.
4. Install lightweight deps (`ipywidgets`, `aria2`, `huggingface_hub[cli]`).
5. Clone this repo (or your fork) to `/content/ComfyUI`.
6. Selector UI — pick pipelines (Flux 2 Dev, Z-Image Turbo 6B, Qwen Image Edit 2511, …) and per-pipeline LoRAs. Click **Launch**.
7. Downloads weights using `hf` (HuggingFace) + `aria2c` (CivitAI), installs `requirements.txt`, starts `main.py`.
8. Starts a `cloudflared` quick tunnel; the `https://*.trycloudflare.com` URL prints inline.

## Customising the repo URL

By default the notebook clones `https://github.com/samsam27/ComfyUI.git`. To use a different repo
(e.g. your own fork or upstream), set the env var before running section 5:

```python
import os
os.environ["COMFYUI_COLAB_REPO_URL"] = "https://github.com/<you>/ComfyUI.git"
```

## Known caveats

- Free-tier Colab GPUs (T4) will run the notebook but Flux 2 Dev (~61 GB) won't fit. Z-Image Turbo 6B is the only pipeline that comfortably fits on a T4.
- The `cloudflared` URL changes every run. For a persistent URL, upgrade to a named Cloudflare Tunnel.
- Clear all cell outputs before sharing the notebook — a `pip install -v` failure could have written a token fragment into output.

## Developing

Helpers live in `colab/launcher_helpers.py` and are unit-tested in `tests-unit/colab_test/`:

```bash
.venv/bin/python -m pytest tests-unit/colab_test/ -v
```
