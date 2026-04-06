#!/usr/bin/env python3
"""
model_manager.py -- Interactive model downloader and lifecycle manager for ComfyUI.

Downloads pipeline models and LoRAs from HuggingFace and CivitAI, symlinks them
into the ComfyUI models directory, and supports cleanup to free disk space.

Usage:
    python scripts/model_manager.py            # Interactive download
    python scripts/model_manager.py --cleanup   # Remove downloaded models + HF cache
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants & Paths
# ---------------------------------------------------------------------------

COMFYUI_DIR = Path(__file__).resolve().parent.parent
# models/ is a symlink -> /home/samsam/models/comfyui
MODELS_DIR = (COMFYUI_DIR / "models").resolve()
REGISTRY_PATH = COMFYUI_DIR / "scripts" / "model_registry.json"
HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"
AI_TOOLKIT_OUTPUT = Path.home() / "ai-toolkit" / "output"
TEMP_MANIFEST = MODELS_DIR / ".tmp_models_manifest.json"
HF_CLI = Path.home() / ".local" / "bin" / "hf"

# ANSI colors
RST = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"
WHITE = "\033[37m"

logging.basicConfig(
    level=logging.INFO,
    format=f"  {DIM}%(asctime)s{RST}  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("model_manager")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_registry() -> dict:
    """Load the model registry JSON."""
    with open(REGISTRY_PATH) as f:
        return json.load(f)


def get_disk_free_gb(path: Path | None = None) -> float:
    """Return free disk space in GB for the filesystem containing path."""
    p = path or MODELS_DIR
    st = os.statvfs(p)
    return (st.f_bavail * st.f_frsize) / (1024**3)


def get_disk_total_gb(path: Path | None = None) -> float:
    """Return total disk space in GB for the filesystem containing path."""
    p = path or MODELS_DIR
    st = os.statvfs(p)
    return (st.f_blocks * st.f_frsize) / (1024**3)


def format_size(gb: float) -> str:
    """Human-readable size string."""
    if gb < 1.0:
        return f"{gb * 1024:.0f} MB"
    return f"{gb:.1f} GB"


def _has_command(name: str) -> bool:
    """Check if a command exists on PATH."""
    return shutil.which(name) is not None


def _run(cmd: list[str], env: dict | None = None, check: bool = True, **kwargs):
    """Run a subprocess with merged environment."""
    full_env = {**os.environ, **(env or {})}
    return subprocess.run(cmd, env=full_env, check=check, **kwargs)


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------


def print_header():
    """Show disk usage header with a colored bar."""
    free = get_disk_free_gb()
    total = get_disk_total_gb()
    used = total - free
    pct_used = used / total if total > 0 else 0

    # Color based on free space
    if free > 100:
        bar_color = GREEN
    elif free > 50:
        bar_color = YELLOW
    else:
        bar_color = RED

    bar_width = 40
    filled = int(bar_width * pct_used)
    empty = bar_width - filled

    bar = f"{bar_color}{'#' * filled}{DIM}{'.' * empty}{RST}"

    print()
    print(f"  {BOLD}{CYAN}Model Manager{RST}")
    print(f"  {DIM}{'=' * 52}{RST}")
    print()
    print(f"  Disk  [{bar}]  {format_size(used)} / {format_size(total)}")
    print(f"  Free  {bar_color}{BOLD}{format_size(free)}{RST}")
    print()


# ---------------------------------------------------------------------------
# Interactive Selection (gum with fallback)
# ---------------------------------------------------------------------------


def _gum_available() -> bool:
    return _has_command("gum")


def select_pipelines(registry: dict) -> list[str]:
    """Interactive multi-select for pipelines."""
    pipelines = registry["pipelines"]
    items = []
    for key, data in pipelines.items():
        label = f"{data['display_name']}  ({format_size(data['total_size_gb'])})"
        items.append((key, label))

    if _gum_available():
        # Build gum choose args
        labels = [f"{key}|{label}" for key, label in items]
        try:
            result = subprocess.run(
                ["gum", "choose", "--no-limit", "--header", "Select pipelines to download:"]
                + labels,
                capture_output=True,
                text=True,
                check=True,
            )
            selected_labels = result.stdout.strip().split("\n")
            return [s.split("|")[0] for s in selected_labels if s.strip()]
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass  # fall through to manual

    # Fallback: numbered input
    print(f"  {BOLD}Select pipelines to download:{RST}")
    print()
    for i, (key, label) in enumerate(items, 1):
        print(f"    {CYAN}{i}{RST}. {label}")
    print()
    raw = input(f"  Enter numbers (comma-separated, e.g. 1,3): ").strip()
    if not raw:
        return []

    selected = []
    for token in raw.split(","):
        token = token.strip()
        if token.isdigit():
            idx = int(token) - 1
            if 0 <= idx < len(items):
                selected.append(items[idx][0])
    return selected


def select_loras(pipeline_key: str, pipeline_data: dict) -> list[dict]:
    """Interactive multi-select for LoRAs within a pipeline."""
    loras = pipeline_data.get("loras", [])
    if not loras:
        return []

    items = []
    for lora in loras:
        label = f"{lora['display_name']}  ({format_size(lora['size_gb'])})"
        items.append((lora, label))

    display_name = pipeline_data["display_name"]

    if _gum_available():
        labels = [f"{i}|{label}" for i, (_, label) in enumerate(items)]
        try:
            result = subprocess.run(
                [
                    "gum",
                    "choose",
                    "--no-limit",
                    "--header",
                    f"Select LoRAs for {display_name}:",
                ]
                + labels,
                capture_output=True,
                text=True,
                check=True,
            )
            selected_labels = result.stdout.strip().split("\n")
            selected_indices = []
            for s in selected_labels:
                if s.strip():
                    idx_str = s.split("|")[0]
                    if idx_str.isdigit():
                        selected_indices.append(int(idx_str))
            return [items[i][0] for i in selected_indices if i < len(items)]
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass

    # Fallback
    print(f"  {BOLD}Select LoRAs for {display_name}:{RST}")
    print()
    for i, (lora, label) in enumerate(items, 1):
        print(f"    {CYAN}{i}{RST}. {label}")
    print(f"    {DIM}0. None{RST}")
    print()
    raw = input(f"  Enter numbers (comma-separated, 0 for none): ").strip()
    if not raw or raw == "0":
        return []

    selected = []
    for token in raw.split(","):
        token = token.strip()
        if token.isdigit():
            idx = int(token) - 1
            if 0 <= idx < len(items):
                selected.append(items[idx][0])
    return selected


# ---------------------------------------------------------------------------
# HuggingFace Cache Inspection
# ---------------------------------------------------------------------------


def _hf_cache_blob_path(repo: str, filename: str) -> Path | None:
    """
    Check if a file exists in the HF cache. Returns the resolved blob path
    if found, or None if not cached.

    HF cache structure:
        ~/.cache/huggingface/hub/models--{org}--{repo}/snapshots/{hash}/{filename}
    The snapshot file is a symlink to a blob in the blobs/ directory.
    """
    repo_dir_name = "models--" + repo.replace("/", "--")
    repo_dir = HF_CACHE / repo_dir_name / "snapshots"

    if not repo_dir.exists():
        return None

    # Check all snapshots (usually just one)
    for snapshot in repo_dir.iterdir():
        candidate = snapshot / filename
        if candidate.exists():
            # Resolve the symlink chain to the actual blob
            resolved = candidate.resolve()
            if resolved.exists():
                return resolved

    return None


# ---------------------------------------------------------------------------
# Download Functions
# ---------------------------------------------------------------------------


def download_hf_model(model: dict, dest_dir: Path, registry: dict) -> Path:
    """
    Download a model from HuggingFace.

    Strategy:
    1. Check if already in HF cache -> symlink if found
    2. Otherwise run `hf download` to populate cache
    3. Symlink from cache to dest_dir

    Returns the final path of the file in dest_dir.
    """
    config = registry["config"]
    repo = model["hf_repo"]
    hf_file = model["hf_file"]
    local_name = model["filename"]
    dest_path = dest_dir / local_name

    # If destination already exists and is valid, skip
    if dest_path.exists() or dest_path.is_symlink():
        if dest_path.is_symlink():
            target = dest_path.resolve()
            if target.exists():
                log.info(f"{GREEN}EXISTS{RST}  {local_name} (symlink)")
                return dest_path
            else:
                # Broken symlink, remove and re-download
                dest_path.unlink()
        else:
            log.info(f"{GREEN}EXISTS{RST}  {local_name} (file)")
            return dest_path

    # 1. Check HF cache
    blob_path = _hf_cache_blob_path(repo, hf_file)

    if blob_path is None:
        # 2. Download via hf CLI
        log.info(f"{CYAN}DOWNLOAD{RST}  {repo} / {hf_file}")

        hf_cmd = str(HF_CLI) if HF_CLI.exists() else "hf"
        cmd = [
            hf_cmd,
            "download",
            repo,
            hf_file,
            *config.get("hf_download_flags", []),
        ]

        env = dict(config.get("hf_env", {}))
        _run(cmd, env=env)

        # Now it should be in cache
        blob_path = _hf_cache_blob_path(repo, hf_file)
        if blob_path is None:
            raise RuntimeError(
                f"Downloaded {repo}/{hf_file} but could not find it in HF cache"
            )

    # 3. Symlink from cache
    # The symlink target should be the snapshot path (which itself links to blob),
    # so we link to the snapshot file, not the blob directly.
    # This matches how ComfyUI discovers files.
    repo_dir_name = "models--" + repo.replace("/", "--")
    snapshot_file = None
    snapshots_dir = HF_CACHE / repo_dir_name / "snapshots"
    if snapshots_dir.exists():
        for snapshot in snapshots_dir.iterdir():
            candidate = snapshot / hf_file
            if candidate.exists():
                snapshot_file = candidate
                break

    link_target = snapshot_file if snapshot_file else blob_path

    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path.symlink_to(link_target)
    log.info(f"{GREEN}SYMLINK{RST}  {local_name} -> {link_target}")

    return dest_path


def download_civitai_model(lora: dict, dest_dir: Path, registry: dict) -> Path:
    """
    Download a model from CivitAI directly to dest_dir.

    Prefers aria2c for parallel downloads, falls back to wget.
    """
    config = registry["config"]
    url = lora["civitai_url"]
    api_key = config.get("civitai_api_key", "")
    local_name = lora["filename"]
    dest_path = dest_dir / local_name

    # If already exists, skip
    if dest_path.exists():
        log.info(f"{GREEN}EXISTS{RST}  {local_name}")
        return dest_path

    dest_dir.mkdir(parents=True, exist_ok=True)

    # Add API key as query parameter
    separator = "&" if "?" in url else "?"
    download_url = f"{url}{separator}token={api_key}"

    if _has_command("aria2c"):
        log.info(f"{CYAN}DOWNLOAD{RST}  {local_name} via aria2c")
        cmd = [
            "aria2c",
            *config.get("aria2c_flags", []),
            f"--dir={dest_dir}",
            f"--out={local_name}",
            download_url,
        ]
        _run(cmd, check=True)
    elif _has_command("wget"):
        log.info(f"{CYAN}DOWNLOAD{RST}  {local_name} via wget")
        cmd = [
            "wget",
            "-q",
            "--show-progress",
            "-O",
            str(dest_path),
            download_url,
        ]
        _run(cmd, check=True)
    else:
        raise RuntimeError(
            "Neither aria2c nor wget found. Install one to download from CivitAI."
        )

    if not dest_path.exists():
        raise RuntimeError(f"Download failed: {local_name} not found at {dest_path}")

    log.info(f"{GREEN}OK{RST}  {local_name} ({format_size(lora['size_gb'])})")
    return dest_path


def download_model_file(model: dict, dest_dir: Path, registry: dict) -> Path:
    """Route to the correct download handler based on source."""
    source = model.get("source", "hf")
    if source == "hf":
        return download_hf_model(model, dest_dir, registry)
    elif source == "civitai":
        return download_civitai_model(model, dest_dir, registry)
    else:
        raise ValueError(f"Unknown source: {source}")


# ---------------------------------------------------------------------------
# Pipeline Download (with progress)
# ---------------------------------------------------------------------------


def _try_import_rich():
    """Try to import rich progress components."""
    try:
        from rich.console import Console
        from rich.progress import (
            BarColumn,
            Progress,
            SpinnerColumn,
            TextColumn,
            TimeElapsedColumn,
        )

        return Console, Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
    except ImportError:
        return None


def download_pipeline(
    pipeline_key: str,
    pipeline_data: dict,
    selected_loras: list[dict],
    registry: dict,
) -> list[dict]:
    """
    Download all models + selected LoRAs for a pipeline.

    Returns a list of file records for the manifest.
    """
    display_name = pipeline_data["display_name"]
    models = pipeline_data["models"]
    all_items = models + selected_loras

    created_files = []

    rich_imports = _try_import_rich()

    if rich_imports:
        Console, Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn = (
            rich_imports
        )
        console = Console()
        console.print(f"\n  [bold cyan]{display_name}[/bold cyan]  ({len(all_items)} files)\n")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=30),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(f"Downloading {display_name}", total=len(all_items))

            for item in all_items:
                progress.update(task, description=f"  {item['filename']}")

                # Determine destination
                if "dest_subdir" in item:
                    dest_dir = MODELS_DIR / item["dest_subdir"]
                else:
                    dest_dir = MODELS_DIR / "loras"

                path = download_model_file(item, dest_dir, registry)
                created_files.append(
                    {
                        "path": str(path),
                        "filename": item["filename"],
                        "pipeline": pipeline_key,
                        "is_symlink": path.is_symlink(),
                        "hf_repo": item.get("hf_repo"),
                        "source": item.get("source", "hf"),
                    }
                )
                progress.advance(task)
    else:
        # Fallback: simple text progress
        print(f"\n  {BOLD}{CYAN}{display_name}{RST}  ({len(all_items)} files)\n")

        for i, item in enumerate(all_items, 1):
            print(f"  [{i}/{len(all_items)}] {item['filename']}")

            if "dest_subdir" in item:
                dest_dir = MODELS_DIR / item["dest_subdir"]
            else:
                dest_dir = MODELS_DIR / "loras"

            path = download_model_file(item, dest_dir, registry)
            created_files.append(
                {
                    "path": str(path),
                    "filename": item["filename"],
                    "pipeline": pipeline_key,
                    "is_symlink": path.is_symlink(),
                    "hf_repo": item.get("hf_repo"),
                    "source": item.get("source", "hf"),
                }
            )

    return created_files


# ---------------------------------------------------------------------------
# AI Toolkit LoRA Symlinks
# ---------------------------------------------------------------------------


def symlink_ai_toolkit_loras() -> list[dict]:
    """
    Scan ~/ai-toolkit/output/ for .safetensors files and symlink them
    into models/loras/. These are NEVER cleaned up.

    Returns a list of file records for the manifest (marked permanent).
    """
    loras_dir = MODELS_DIR / "loras"
    loras_dir.mkdir(parents=True, exist_ok=True)

    if not AI_TOOLKIT_OUTPUT.exists():
        log.info(f"{DIM}AI Toolkit output not found at {AI_TOOLKIT_OUTPUT}{RST}")
        return []

    ai_toolkit_files = []
    found = 0

    for dirpath, _dirnames, filenames in os.walk(AI_TOOLKIT_OUTPUT):
        for fname in filenames:
            if not fname.endswith(".safetensors"):
                continue

            src = Path(dirpath) / fname
            dest = loras_dir / fname

            if dest.is_symlink():
                # Already linked -- check if target matches
                if dest.resolve() == src.resolve():
                    ai_toolkit_files.append(
                        {
                            "path": str(dest),
                            "filename": fname,
                            "source": "ai_toolkit",
                            "permanent": True,
                        }
                    )
                    found += 1
                    continue
                else:
                    dest.unlink()
            elif dest.exists():
                # Real file with same name -- don't overwrite
                continue

            dest.symlink_to(src)
            ai_toolkit_files.append(
                {
                    "path": str(dest),
                    "filename": fname,
                    "source": "ai_toolkit",
                    "permanent": True,
                }
            )
            found += 1

    if found > 0:
        log.info(f"{GREEN}AI Toolkit{RST}  {found} LoRAs symlinked")

    return ai_toolkit_files


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def save_manifest(created_files: list[dict], ai_toolkit_files: list[dict]):
    """
    Save a JSON manifest of:
    - temporary files (for cleanup -- pipeline models + LoRAs)
    - permanent files (AI Toolkit LoRAs -- never cleaned)
    - HF cache repos used (for cache cleanup)
    """
    hf_repos_used = set()
    for f in created_files:
        repo = f.get("hf_repo")
        if repo:
            hf_repos_used.add(repo)

    manifest = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "temporary_files": [f for f in created_files if not f.get("permanent")],
        "permanent_files": ai_toolkit_files,
        "hf_cache_repos": sorted(hf_repos_used),
    }

    with open(TEMP_MANIFEST, "w") as f:
        json.dump(manifest, f, indent=2)

    log.info(f"{GREEN}MANIFEST{RST}  Saved to {TEMP_MANIFEST}")


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------


def cleanup(keep_cache: bool = False):
    """
    Read manifest and delete temporary files.

    - Removes symlinks and direct-downloaded files
    - Optionally deletes HF cache repos to free real disk space
    - Never touches permanent (AI Toolkit) files
    """
    if not TEMP_MANIFEST.exists():
        print(f"  {YELLOW}No manifest found at {TEMP_MANIFEST}{RST}")
        print(f"  {DIM}Nothing to clean up.{RST}")
        return

    with open(TEMP_MANIFEST) as f:
        manifest = json.load(f)

    temp_files = manifest.get("temporary_files", [])
    hf_repos = manifest.get("hf_cache_repos", [])

    freed_count = 0
    freed_bytes = 0
    errors = []

    print(f"\n  {BOLD}Cleaning up temporary model files...{RST}\n")

    for record in temp_files:
        path = Path(record["path"])
        fname = record.get("filename", path.name)

        if not path.exists() and not path.is_symlink():
            continue

        try:
            if path.is_symlink():
                # For symlinks, the actual space is in HF cache (handled below)
                path.unlink()
                log.info(f"{RED}REMOVED{RST}  {fname} (symlink)")
                freed_count += 1
            else:
                # Direct file (CivitAI downloads)
                size = path.stat().st_size
                path.unlink()
                freed_bytes += size
                freed_count += 1
                log.info(f"{RED}REMOVED{RST}  {fname} ({format_size(size / (1024**3))})")
        except OSError as e:
            errors.append(f"{fname}: {e}")

    # Delete HF cache repos to free real disk space
    cache_freed = 0
    if not keep_cache and hf_repos:
        print(f"\n  {BOLD}Cleaning HF cache repos...{RST}\n")
        for repo in hf_repos:
            repo_dir_name = "models--" + repo.replace("/", "--")
            repo_dir = HF_CACHE / repo_dir_name
            if repo_dir.exists():
                # Calculate size before deletion
                repo_size = sum(
                    f.stat().st_size
                    for f in repo_dir.rglob("*")
                    if f.is_file() and not f.is_symlink()
                )
                try:
                    shutil.rmtree(repo_dir)
                    cache_freed += repo_size
                    log.info(
                        f"{RED}CACHE{RST}  Removed {repo_dir_name} "
                        f"({format_size(repo_size / (1024**3))})"
                    )
                except OSError as e:
                    errors.append(f"Cache {repo}: {e}")

    # Remove manifest
    TEMP_MANIFEST.unlink(missing_ok=True)

    # Summary
    total_freed = freed_bytes + cache_freed
    print()
    print(f"  {BOLD}Cleanup Summary{RST}")
    print(f"  {'=' * 40}")
    print(f"  Files removed:  {freed_count}")
    print(f"  Space freed:    {format_size(total_freed / (1024**3))}")
    if cache_freed > 0:
        print(f"    (cache):      {format_size(cache_freed / (1024**3))}")
    if errors:
        print(f"  {RED}Errors:         {len(errors)}{RST}")
        for e in errors:
            print(f"    {RED}- {e}{RST}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    """Interactive model download workflow."""
    print_header()

    registry = load_registry()

    # 1. Symlink AI Toolkit LoRAs (always)
    ai_toolkit_files = symlink_ai_toolkit_loras()

    # 2. Select pipelines
    selected_keys = select_pipelines(registry)
    if not selected_keys:
        print(f"\n  {YELLOW}No pipelines selected. Exiting.{RST}\n")
        return

    # 3. Select LoRAs for each pipeline
    pipeline_loras: dict[str, list[dict]] = {}
    for key in selected_keys:
        pipeline_data = registry["pipelines"][key]
        loras = select_loras(key, pipeline_data)
        pipeline_loras[key] = loras

    # 4. Estimate total download size and check disk space
    total_needed_gb = 0.0
    for key in selected_keys:
        pipeline_data = registry["pipelines"][key]
        # Base models
        for model in pipeline_data["models"]:
            dest = MODELS_DIR / model["dest_subdir"] / model["filename"]
            if not dest.exists():
                total_needed_gb += model["size_gb"]
        # Selected LoRAs
        for lora in pipeline_loras[key]:
            dest = MODELS_DIR / "loras" / lora["filename"]
            if not dest.exists():
                total_needed_gb += lora["size_gb"]

    free_gb = get_disk_free_gb()

    print()
    print(f"  {BOLD}Download Plan{RST}")
    print(f"  {'=' * 40}")
    for key in selected_keys:
        pd = registry["pipelines"][key]
        n_loras = len(pipeline_loras[key])
        lora_str = f" + {n_loras} LoRAs" if n_loras > 0 else ""
        print(f"    {pd['display_name']}{lora_str}")
    print()
    print(f"  Estimated new downloads: {BOLD}{format_size(total_needed_gb)}{RST}")
    print(f"  Free disk space:         {format_size(free_gb)}")

    if total_needed_gb > free_gb * 0.9:
        print(
            f"\n  {RED}{BOLD}WARNING:{RST} {RED}Download size may exceed available "
            f"disk space!{RST}"
        )
        try:
            answer = input(f"  Continue anyway? [y/N]: ").strip().lower()
            if answer != "y":
                print(f"\n  {YELLOW}Aborted.{RST}\n")
                return
        except (EOFError, KeyboardInterrupt):
            print(f"\n  {YELLOW}Aborted.{RST}\n")
            return
    print()

    # 5. Download each pipeline
    all_created: list[dict] = []
    for key in selected_keys:
        pipeline_data = registry["pipelines"][key]
        loras = pipeline_loras[key]
        created = download_pipeline(key, pipeline_data, loras, registry)
        all_created.extend(created)

    # 6. Save manifest
    save_manifest(all_created, ai_toolkit_files)

    # 7. Summary
    print()
    print(f"  {GREEN}{BOLD}Done!{RST}")
    print(f"  {'=' * 40}")
    print(f"  Files:  {len(all_created)} downloaded/linked")
    symlinks = sum(1 for f in all_created if f.get("is_symlink"))
    if symlinks:
        print(f"  Symlinks: {symlinks} (saving disk space via HF cache)")
    print(f"  Free:   {format_size(get_disk_free_gb())}")
    print()
    print(f"  {DIM}Run with --cleanup to remove downloaded models{RST}")
    print()


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if "--cleanup" in sys.argv:
        keep_cache = "--keep-cache" in sys.argv
        cleanup(keep_cache=keep_cache)
    else:
        try:
            main()
        except KeyboardInterrupt:
            print(f"\n\n  {YELLOW}Interrupted.{RST}\n")
            sys.exit(130)
