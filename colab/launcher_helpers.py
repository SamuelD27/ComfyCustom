"""Pure-Python helpers for the Colab ComfyUI launcher.

All functions here must be importable in an environment with only the
Python stdlib — heavy imports (huggingface_hub, ipywidgets, nbformat,
google.colab) go inside functions so tests run without them installed.
"""

__all__ = ["__version__"]

__version__ = "0.1.0"

import re
import subprocess

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

__all__ += ["TorchDecision", "torch_action"]


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
