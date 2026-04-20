# ComfyUI on Colab — Personalised Launcher

One-click launch of this repo's ComfyUI setup on Google Colab, with registry-driven
pipeline selection, on-demand weight downloads, and a public `cloudflared` URL.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SamuelD27/ComfyCustom/blob/Collab/colab/ComfyUI_Colab.ipynb)

The launcher lives on the long-lived `Collab` branch of `SamuelD27/ComfyCustom` (parallel to
`main`) so the Colab-specific code never merges into the main repo. If you fork this repo,
update the badge link: replace `SamuelD27/ComfyCustom` with `<your-gh-user>/<repo>` and
adjust the branch name to match wherever you store the Colab files.

## Prerequisites

- **Colab Pro+** (A100/G4 access and background execution).
- **Drive permission** — prompted on first run; used only for a ~1 KB `prefs.json`.
- **Secrets** added to Colab's key-icon sidebar (optional but recommended):
  - `HF_TOKEN` — HuggingFace access token
  - `CIVITAI_TOKEN` — CivitAI API key (for LoRA downloads)

If a secret is missing, the notebook falls back to a hidden `getpass` prompt in the cell.

## How it works

The notebook has three cells:

1. **Cell 1 — Setup + selector.** Hardcoded tokens, GPU detect + conditional cu128 torch reinstall, clone repo to `/content/ComfyUI`, render pipeline + LoRA checkboxes. Click **Save selection**.
2. **Cell 2 — Install + download + launch.** `uv pip` installs `requirements.txt` + enabled custom node reqs, `hf` + `aria2c` pull only the selected weights, workflow JSONs in `colab/workflows/` are copied into `user/default/workflows/`, `main.py` starts on port 8188, `cloudflared` prints the public `https://*.trycloudflare.com` URL.
3. **Cell 3 — Restart.** Kills the running ComfyUI and relaunches it. No re-download, no re-install; the cloudflared tunnel stays up and the public URL is unchanged. Use this after editing workflows locally and pushing, or when ComfyUI crashed.

### Why Cell 3 exists

Cell 2 is expensive (≥ 15 min the first time) because of model downloads and dep installs. Cell 3 reuses everything on disk, so a restart takes ~10 s and keeps the same public URL. It also `git pull`s `colab/workflows/` by default (disable with `os.environ["COMFYUI_RESTART_PULL"] = "0"` before running it).

### Workflows

Workflow JSONs live in `colab/workflows/` inside the repo. ComfyUI's own `user/default/workflows/` is gitignored, so the launcher mirrors `colab/workflows/ → user/default/workflows/` on each run of Cell 2 or Cell 3. To add or edit a workflow, drop the JSON into `colab/workflows/` (mirroring the subdirectory layout), commit, push, and re-run Cell 3.

## Customising the repo URL

By default the notebook clones `https://github.com/SamuelD27/ComfyCustom.git` at branch `Collab`.
To use a different repo or branch, set env vars before running section 5:

```python
import os
os.environ["COMFYUI_COLAB_REPO_URL"] = "https://github.com/<you>/<repo>.git"
os.environ["COMFYUI_COLAB_BRANCH"] = "<branch-with-colab-dir>"
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
