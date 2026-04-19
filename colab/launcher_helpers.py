"""Pure-Python helpers for the Colab ComfyUI launcher.

All functions here must be importable in an environment with only the
Python stdlib — heavy imports (huggingface_hub, ipywidgets, nbformat,
google.colab) go inside functions so tests run without them installed.
"""

__all__ = ["__version__"]

__version__ = "0.1.0"

import getpass as _getpass
import os
import re
import subprocess
import sys

__all__ += ["compute_cap_from_smi", "compute_cap_from_name"]


def compute_cap_from_smi() -> int:
    """Return compute capability as an integer (e.g. 80 for sm_80).

    Returns 0 if nvidia-smi is unavailable or fails for any reason.
    """
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return 0
    first = out.strip().splitlines()[0] if out.strip() else ""
    if "." not in first:
        return 0
    major, minor = first.split(".", 1)
    try:
        return int(major) * 10 + int(minor)
    except ValueError:
        return 0


_NAME_TO_CC = [
    (re.compile(r"a100", re.I), 80),
    (re.compile(r"h100|h200", re.I), 90),
    (re.compile(r"rtx\s*(50|pro\s*6000)|blackwell|b200|b100", re.I), 120),
    (re.compile(r"l4\b|l40", re.I), 89),
    (re.compile(r"\bt4\b|tesla\s*t4", re.I), 75),
]


def compute_cap_from_name(name: str) -> int:
    """Map a GPU name string to compute capability. Returns 0 if unknown."""
    for pat, cc in _NAME_TO_CC:
        if pat.search(name):
            return cc
    return 0


from dataclasses import dataclass, field
from typing import List

__all__ += ["TorchDecision", "torch_action", "load_secret", "apply_hf_env"]


@dataclass
class TorchDecision:
    action: str  # "keep" or "reinstall"
    reason: str
    pip_args: List[str] = field(default_factory=list)
    index_url: str = ""


_CU128_PIP_ARGS = [
    "torch==2.11.0",
    "torchvision==0.26.0",
    "torchaudio==2.11.0",
]
_CU128_INDEX = "https://download.pytorch.org/whl/cu128"


def torch_action(cc: int, arch_list: List[str]) -> TorchDecision:
    """Decide whether to keep Colab's preinstalled torch or reinstall cu128.

    Args:
        cc: compute capability as integer (0 = unknown / no GPU).
        arch_list: output of ``torch.cuda.get_arch_list()``.
    """
    if cc == 0:
        return TorchDecision(
            action="keep",
            reason="No GPU detected — keeping preinstalled torch.",
        )
    target = f"sm_{cc}"
    if target in arch_list:
        return TorchDecision(
            action="keep",
            reason=f"Preinstalled torch already supports {target}.",
        )
    return TorchDecision(
        action="reinstall",
        reason=(
            f"{target} not in preinstalled arch list {arch_list}; "
            "reinstalling torch from cu128 index (broader third-party wheel coverage)."
        ),
        pip_args=list(_CU128_PIP_ARGS),
        index_url=_CU128_INDEX,
    )


def load_secret(name: str, interactive: bool = True) -> "str | None":
    """Load a secret (HF_TOKEN, CIVITAI_TOKEN, etc.) from Colab userdata
    or fall back to an interactive getpass prompt.

    Returns None if not available and interactive=False.
    """
    mod = sys.modules.get("google.colab")
    if mod is None:
        try:
            import google.colab as mod  # type: ignore
        except ImportError:
            mod = None
    if mod is not None and hasattr(mod, "userdata"):
        ud = mod.userdata
        try:
            val = ud.get(name)
            if val:
                return val
        except getattr(ud, "SecretNotFoundError", Exception):
            pass
        except getattr(ud, "NotebookAccessError", Exception):
            print(
                f"[secrets] '{name}' exists but notebook is not authorised "
                "— toggle access in the key icon on the left sidebar."
            )
            return None
    if interactive:
        try:
            val = _getpass.getpass(f"{name} (hidden, or press Enter to skip): ")
        except Exception:
            return None
        return val or None
    return None


def apply_hf_env(token: "str | None") -> None:
    """Set both HF env var names that huggingface_hub/`hf` CLI look up."""
    if not token:
        return
    os.environ["HF_TOKEN"] = token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = token


import json
import tempfile
from pathlib import Path
from typing import Any, Dict

__all__ += ["load_registry", "list_pipelines", "pipeline_models", "pipeline_loras"]


def load_registry(path: "str | Path") -> Dict[str, Any]:
    """Load scripts/model_registry.json into a dict."""
    with open(path) as f:
        return json.load(f)


def list_pipelines(registry: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return a flat list of pipeline summaries for the selector UI."""
    out = []
    for key, pipe in registry.get("pipelines", {}).items():
        out.append(
            {
                "key": key,
                "display_name": pipe.get("display_name", key),
                "total_size_gb": pipe.get("total_size_gb", 0),
                "model_count": len(pipe.get("models", [])),
                "lora_count": len(pipe.get("loras", [])),
            }
        )
    return out


def pipeline_models(registry: Dict[str, Any], key: str) -> List[Dict[str, Any]]:
    """Return the model spec list for a pipeline. Raises KeyError if unknown."""
    return list(registry["pipelines"][key].get("models", []))


def pipeline_loras(registry: Dict[str, Any], key: str) -> List[Dict[str, Any]]:
    """Return the lora spec list for a pipeline.

    Returns an empty list if the pipeline has no loras defined.
    Raises KeyError if the pipeline key is not in the registry.
    """
    return list(registry["pipelines"][key].get("loras", []))


__all__ += ["load_prefs", "save_prefs"]


def load_prefs(path: "str | Path") -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        with open(p) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def save_prefs(path: "str | Path", prefs: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(prefs, f, indent=2)


__all__ += ["write_auth_header"]


def write_auth_header(token: "str | None", tmp_dir: "str | Path") -> "Path | None":
    """Write an 'Authorization: Bearer <token>' header file (mode 0600).

    Returns the Path of the file, or None if no token was supplied.
    """
    if not token:
        return None
    d = Path(tmp_dir)
    d.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix="auth_", suffix=".hdr", dir=str(d))
    try:
        with os.fdopen(fd, "w") as f:
            f.write(f"Authorization: Bearer {token}\n")
    except Exception:
        try:
            os.unlink(name)
        finally:
            raise
    os.chmod(name, 0o600)
    return Path(name)


__all__ += ["build_hf_download_cmd", "build_aria2c_cmd"]


def build_hf_download_cmd(
    repo: str,
    filename: str,
    dest_dir: str,
    max_workers: int = 32,
) -> List[str]:
    """Build argv for HuggingFace CLI `hf download` command.

    Args:
        repo: HuggingFace repo ID (e.g. "black-forest-labs/FLUX.2-dev").
        filename: File to download from the repo.
        dest_dir: Destination directory for the downloaded file.
        max_workers: Number of parallel download workers (default 32).

    Returns:
        List of command arguments ready for subprocess.run().
    """
    return [
        "hf",
        "download",
        repo,
        filename,
        "--local-dir",
        dest_dir,
        "--max-workers",
        str(max_workers),
    ]


def build_aria2c_cmd(
    url: str,
    dest_dir: str,
    filename: str,
    auth_header_file: "str | None" = None,
    connections: int = 16,
) -> List[str]:
    """Build argv for aria2c download command.

    Configures aria2c for fast parallel downloads with optional authentication.

    Args:
        url: Full download URL (HTTP/HTTPS).
        dest_dir: Destination directory for the downloaded file.
        filename: Output filename.
        auth_header_file: Path to file containing "Authorization: Bearer <token>"
                         (created by write_auth_header). Optional.
        connections: Number of parallel connections (default 16).

    Returns:
        List of command arguments ready for subprocess.run().
    """
    cmd = [
        "aria2c",
        "-x", str(connections),
        "-s", str(connections),
        "-d", dest_dir,
        "-o", filename,
        "--continue=true",
        "--auto-file-renaming=false",
        "--allow-overwrite=true",
        "--console-log-level=warn",
        "--quiet=true",
    ]
    if auth_header_file:
        cmd.append(f"--header=@{auth_header_file}")
    cmd.append(url)
    return cmd
