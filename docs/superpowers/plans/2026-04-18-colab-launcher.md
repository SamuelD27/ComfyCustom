# Colab Launcher Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a self-contained Colab notebook that clones this ComfyUI repo, auto-detects the allocated GPU, installs deps, presents an ipywidgets pipeline/LoRA selector driven by the existing `scripts/model_registry.json`, downloads weights on demand, and exposes ComfyUI over a cloudflared quick tunnel — with zero interactive friction on Colab Pro+ A100/G4 sessions.

**Architecture:** Registry-based (Option A). Single `master` clone; pipeline selection reads the existing `scripts/model_registry.json`. Notebook is a thin orchestration layer over a testable pure-Python helper module at `colab/launcher_helpers.py`. Cloudflared quick tunnel (no account) for the public URL. ipywidgets for UI. Drive mount is only for a ~1 KB `prefs.json` file (no model caching). Sets of cells: (1) mount Drive, (2) GPU detect, (3) secrets, (4) deps install, (5) clone repo, (6) selector UI, (7) launch callback (install reqs + download + start `main.py`), (8) tunnel.

**Tech Stack:** Python 3.11+, `nbformat` (for notebook construction), `ipywidgets`, `huggingface_hub` + `hf` CLI, `aria2c`, `cloudflared`, pytest + `unittest.mock` for helpers.

**Prerequisites / Assumptions:**
- The notebook clones from a git URL configurable via the `REPO_URL` env var or widget default. Default placeholder: `https://github.com/samsam27/ComfyUI.git`. Sam creates/updates his fork before shipping; the notebook itself has no fork dependency.
- Colab Pro+ is assumed for A100/G4 allocation and background execution — not checked in code.
- `--highvram` is NOT forced; ComfyUI auto-picks. Colab allocations range 22 GB (L4) to 96 GB (G4), and auto-management is safer than a fixed flag.
- All custom nodes under `custom_nodes/` are installed unconditionally in v1. Per-pipeline pruning is out of scope for this plan.

**Files:**
- Create `colab/__init__.py`
- Create `colab/launcher_helpers.py` — all testable logic (built incrementally across Tasks 2–9)
- Create `colab/ComfyUI_Colab.ipynb` — the notebook (scaffolded Task 10, cells filled Tasks 11–13)
- Create `colab/README.md` — Open-in-Colab instructions (Task 14)
- Create `tests-unit/colab_test/__init__.py`
- Create `tests-unit/colab_test/test_launcher_helpers.py`
- Read-only (not modified by this plan): `scripts/model_registry.json`

---

### Task 1: Scaffold `colab/` and `tests-unit/colab_test/`

**Files:**
- Create: `colab/__init__.py`
- Create: `colab/launcher_helpers.py`
- Create: `tests-unit/colab_test/__init__.py`
- Create: `tests-unit/colab_test/test_launcher_helpers.py`

- [ ] **Step 1: Create empty package init files**

```bash
mkdir -p colab tests-unit/colab_test
: > colab/__init__.py
: > tests-unit/colab_test/__init__.py
```

- [ ] **Step 2: Create helper module with a version stub**

Write `colab/launcher_helpers.py`:

```python
"""Pure-Python helpers for the Colab ComfyUI launcher.

All functions here must be importable in an environment with only the
Python stdlib — heavy imports (huggingface_hub, ipywidgets, nbformat,
google.colab) go inside functions so tests run without them installed.
"""

__all__ = ["__version__"]

__version__ = "0.1.0"
```

- [ ] **Step 3: Write a smoke test that proves the harness runs**

Write `tests-unit/colab_test/test_launcher_helpers.py`:

```python
from colab import launcher_helpers


def test_version_is_set():
    assert launcher_helpers.__version__ == "0.1.0"
```

- [ ] **Step 4: Run the test**

Run: `.venv/bin/python -m pytest tests-unit/colab_test/ -v`
Expected: `1 passed`.

- [ ] **Step 5: Commit**

```bash
git add colab/__init__.py colab/launcher_helpers.py tests-unit/colab_test/
git commit -m "chore(colab): scaffold launcher helper module and test package"
```

---

### Task 2: GPU compute-capability detection

**Why this function:** nvidia-smi is present on every Colab runtime at session start, even before any pip install. Detecting compute capability (sm_80, sm_90, sm_120, etc.) drives the torch-reinstall decision in Task 3.

**Files:**
- Modify: `colab/launcher_helpers.py` (append new functions)
- Modify: `tests-unit/colab_test/test_launcher_helpers.py` (append tests)

- [ ] **Step 1: Write failing tests**

Append to `tests-unit/colab_test/test_launcher_helpers.py`:

```python
from unittest.mock import patch
import subprocess

from colab.launcher_helpers import compute_cap_from_smi, compute_cap_from_name


def test_compute_cap_from_smi_parses_standard_output():
    fake = "8.0\n"
    with patch("subprocess.check_output", return_value=fake):
        assert compute_cap_from_smi() == 80


def test_compute_cap_from_smi_handles_multiple_gpus_picks_first():
    fake = "9.0\n7.5\n"
    with patch("subprocess.check_output", return_value=fake):
        assert compute_cap_from_smi() == 90


def test_compute_cap_from_smi_returns_zero_on_failure():
    with patch(
        "subprocess.check_output",
        side_effect=subprocess.CalledProcessError(1, "nvidia-smi"),
    ):
        assert compute_cap_from_smi() == 0


def test_compute_cap_from_smi_returns_zero_when_nvidia_smi_missing():
    with patch("subprocess.check_output", side_effect=FileNotFoundError):
        assert compute_cap_from_smi() == 0


def test_compute_cap_from_name_maps_known_gpus():
    assert compute_cap_from_name("NVIDIA A100-SXM4-40GB") == 80
    assert compute_cap_from_name("NVIDIA H100 80GB HBM3") == 90
    assert compute_cap_from_name("NVIDIA H200") == 90
    assert compute_cap_from_name("NVIDIA RTX PRO 6000 Blackwell") == 120
    assert compute_cap_from_name("NVIDIA L4") == 89
    assert compute_cap_from_name("NVIDIA L40S") == 89
    assert compute_cap_from_name("Tesla T4") == 75
    assert compute_cap_from_name("Some Unknown GPU") == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests-unit/colab_test/ -v`
Expected: `ImportError: cannot import name 'compute_cap_from_smi'` (or similar).

- [ ] **Step 3: Implement the helpers**

Append to `colab/launcher_helpers.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests-unit/colab_test/ -v`
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add colab/launcher_helpers.py tests-unit/colab_test/test_launcher_helpers.py
git commit -m "feat(colab): add GPU compute-capability detection helpers"
```

---

### Task 3: Torch reinstall decision

**Why:** Colab preinstalls `torch 2.10+cu130` with sm_80/sm_90/sm_120 wheels baked in. We only reinstall when the allocated GPU isn't covered. When we do reinstall, cu128 is preferred over cu130 because third-party wheels (xformers, flash-attn, SageAttention) lag on cu130 as of April 2026.

**Files:**
- Modify: `colab/launcher_helpers.py`
- Modify: `tests-unit/colab_test/test_launcher_helpers.py`

- [ ] **Step 1: Write failing tests**

Append to `tests-unit/colab_test/test_launcher_helpers.py`:

```python
from colab.launcher_helpers import torch_action


def test_torch_action_keep_when_cc_in_archlist():
    decision = torch_action(cc=80, arch_list=["sm_80", "sm_90", "sm_120"])
    assert decision.action == "keep"
    assert decision.reason


def test_torch_action_keep_when_cc_is_zero_cpu_only_runtime():
    decision = torch_action(cc=0, arch_list=["sm_80"])
    assert decision.action == "keep"


def test_torch_action_reinstall_when_cc_missing():
    decision = torch_action(cc=120, arch_list=["sm_80", "sm_90"])
    assert decision.action == "reinstall"
    assert "cu128" in decision.reason


def test_torch_action_reinstall_returns_pip_args():
    decision = torch_action(cc=120, arch_list=["sm_80"])
    assert decision.pip_args[0].startswith("torch==")
    assert decision.index_url.endswith("/cu128")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests-unit/colab_test/ -v`
Expected: `ImportError: cannot import name 'torch_action'`.

- [ ] **Step 3: Implement**

Append to `colab/launcher_helpers.py`:

```python
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
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests-unit/colab_test/ -v`
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add colab/launcher_helpers.py tests-unit/colab_test/test_launcher_helpers.py
git commit -m "feat(colab): add torch reinstall decision helper"
```

---

### Task 4: Secrets loader (`HF_TOKEN`, `CIVITAI_TOKEN`)

**Why:** Tokens come from `google.colab.userdata` (preferred) or `getpass` (fallback). They must be set into env vars so `hf` CLI picks them up implicitly; we also write `Authorization` header files for `aria2c` in Task 7 — never on the command line.

**Files:**
- Modify: `colab/launcher_helpers.py`
- Modify: `tests-unit/colab_test/test_launcher_helpers.py`

- [ ] **Step 1: Write failing tests**

Append to `tests-unit/colab_test/test_launcher_helpers.py`:

```python
import os
from unittest.mock import patch

from colab.launcher_helpers import load_secret, apply_hf_env


def test_load_secret_prefers_userdata(monkeypatch):
    class FakeSecretNotFound(Exception):
        pass

    class FakeUserdata:
        SecretNotFoundError = FakeSecretNotFound
        NotebookAccessError = RuntimeError

        @staticmethod
        def get(name):
            return "hf_fake_token_abc"

    with patch.dict("sys.modules", {"google.colab": type("m", (), {"userdata": FakeUserdata})}):
        assert load_secret("HF_TOKEN") == "hf_fake_token_abc"


def test_load_secret_returns_none_when_secret_missing(monkeypatch):
    class FakeSecretNotFound(Exception):
        pass

    class FakeUserdata:
        SecretNotFoundError = FakeSecretNotFound
        NotebookAccessError = RuntimeError

        @staticmethod
        def get(name):
            raise FakeSecretNotFound()

    with patch.dict("sys.modules", {"google.colab": type("m", (), {"userdata": FakeUserdata})}):
        assert load_secret("HF_TOKEN", interactive=False) is None


def test_load_secret_falls_back_to_getpass_when_no_colab(monkeypatch):
    # Simulate non-Colab environment: google.colab not importable
    monkeypatch.setitem(__import__("sys").modules, "google.colab", None)
    with patch("getpass.getpass", return_value="hf_from_prompt"):
        assert load_secret("HF_TOKEN", interactive=True) == "hf_from_prompt"


def test_apply_hf_env_sets_both_variables(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    apply_hf_env("hf_xyz")
    assert os.environ["HF_TOKEN"] == "hf_xyz"
    assert os.environ["HUGGING_FACE_HUB_TOKEN"] == "hf_xyz"


def test_apply_hf_env_noop_on_none(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    apply_hf_env(None)
    assert "HF_TOKEN" not in os.environ
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests-unit/colab_test/ -v`

- [ ] **Step 3: Implement**

Append to `colab/launcher_helpers.py`:

```python
import getpass as _getpass
import os
import sys

__all__ += ["load_secret", "apply_hf_env"]


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
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests-unit/colab_test/ -v`
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add colab/launcher_helpers.py tests-unit/colab_test/test_launcher_helpers.py
git commit -m "feat(colab): add secrets loader with userdata/getpass fallback"
```

---

### Task 5: Registry loader + pipeline view

**Why:** The notebook reads the existing `scripts/model_registry.json` (same file the Bash `model_manager.py` uses). This helper exposes a typed view: list pipelines with `key → display_name, total_size_gb, model_count, loras`. The UI in Task 12 consumes it.

**Files:**
- Modify: `colab/launcher_helpers.py`
- Modify: `tests-unit/colab_test/test_launcher_helpers.py`
- Create: `tests-unit/colab_test/fixtures/registry_sample.json`

- [ ] **Step 1: Create a fixture registry file**

Write `tests-unit/colab_test/fixtures/registry_sample.json`:

```json
{
  "config": {},
  "pipelines": {
    "flux2-dev": {
      "display_name": "Flux 2 Dev",
      "total_size_gb": 66,
      "models": [
        {"filename": "flux2-dev.safetensors", "dest_subdir": "diffusion_models", "size_gb": 61, "source": "hf", "hf_repo": "black-forest-labs/FLUX.2-dev", "hf_file": "flux2-dev.safetensors"}
      ],
      "loras": [
        {"filename": "lora-a.safetensors", "display_name": "LoRA A", "size_gb": 0.5, "source": "hf", "hf_repo": "x/y", "hf_file": "lora-a.safetensors"}
      ]
    },
    "z-image-turbo-6b": {
      "display_name": "Z-Image Turbo 6B",
      "total_size_gb": 21,
      "models": [
        {"filename": "z_image_turbo_bf16.safetensors", "dest_subdir": "diffusion_models", "size_gb": 12.3, "source": "hf", "hf_repo": "Comfy-Org/z-image", "hf_file": "z_image_turbo_bf16.safetensors"}
      ],
      "loras": []
    }
  }
}
```

- [ ] **Step 2: Write failing tests**

Append to `tests-unit/colab_test/test_launcher_helpers.py`:

```python
from pathlib import Path

from colab.launcher_helpers import (
    load_registry,
    list_pipelines,
    pipeline_models,
    pipeline_loras,
)

FIXTURE = Path(__file__).parent / "fixtures" / "registry_sample.json"


def test_load_registry_returns_dict():
    reg = load_registry(FIXTURE)
    assert "pipelines" in reg
    assert "flux2-dev" in reg["pipelines"]


def test_list_pipelines_returns_display_view():
    reg = load_registry(FIXTURE)
    view = list_pipelines(reg)
    assert len(view) == 2
    keys = {p["key"] for p in view}
    assert keys == {"flux2-dev", "z-image-turbo-6b"}
    flux = next(p for p in view if p["key"] == "flux2-dev")
    assert flux["display_name"] == "Flux 2 Dev"
    assert flux["total_size_gb"] == 66
    assert flux["lora_count"] == 1


def test_pipeline_models_returns_list_of_dicts():
    reg = load_registry(FIXTURE)
    models = pipeline_models(reg, "flux2-dev")
    assert len(models) == 1
    assert models[0]["filename"] == "flux2-dev.safetensors"


def test_pipeline_loras_returns_empty_for_pipelines_without_loras():
    reg = load_registry(FIXTURE)
    assert pipeline_loras(reg, "z-image-turbo-6b") == []


def test_pipeline_models_raises_on_unknown_pipeline():
    reg = load_registry(FIXTURE)
    import pytest as _pytest
    with _pytest.raises(KeyError):
        pipeline_models(reg, "does-not-exist")
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests-unit/colab_test/ -v`

- [ ] **Step 4: Implement**

Append to `colab/launcher_helpers.py`:

```python
import json
from pathlib import Path
from typing import Any, Dict, List

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
    """Return the lora spec list for a pipeline. Empty list if none. Raises KeyError if unknown."""
    return list(registry["pipelines"][key].get("loras", []))
```

- [ ] **Step 5: Run tests**

Run: `.venv/bin/python -m pytest tests-unit/colab_test/ -v`

- [ ] **Step 6: Commit**

```bash
git add colab/launcher_helpers.py tests-unit/colab_test/
git commit -m "feat(colab): add model_registry loader and pipeline view helpers"
```

---

### Task 6: Preferences persistence (Drive `prefs.json`)

**Why:** The selector should remember Sam's last choice across sessions. Drive is mounted only for this tiny file (no model caching). Schema: `{"pipelines": ["flux2-dev"], "loras": {"flux2-dev": ["NiceGirls Flux2"]}, "repo_url": "..."}`.

**Files:**
- Modify: `colab/launcher_helpers.py`
- Modify: `tests-unit/colab_test/test_launcher_helpers.py`

- [ ] **Step 1: Write failing tests**

Append to `tests-unit/colab_test/test_launcher_helpers.py`:

```python
def test_load_prefs_returns_empty_dict_for_missing_file(tmp_path):
    from colab.launcher_helpers import load_prefs
    assert load_prefs(tmp_path / "nope.json") == {}


def test_load_prefs_reads_existing_file(tmp_path):
    from colab.launcher_helpers import load_prefs
    p = tmp_path / "prefs.json"
    p.write_text('{"pipelines": ["flux2-dev"], "loras": {}}')
    assert load_prefs(p) == {"pipelines": ["flux2-dev"], "loras": {}}


def test_load_prefs_returns_empty_on_invalid_json(tmp_path):
    from colab.launcher_helpers import load_prefs
    p = tmp_path / "prefs.json"
    p.write_text("not json {")
    assert load_prefs(p) == {}


def test_save_prefs_creates_parent_dir(tmp_path):
    from colab.launcher_helpers import save_prefs, load_prefs
    target = tmp_path / "deep" / "path" / "prefs.json"
    save_prefs(target, {"pipelines": ["z-image-turbo-6b"]})
    assert load_prefs(target) == {"pipelines": ["z-image-turbo-6b"]}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests-unit/colab_test/ -v`

- [ ] **Step 3: Implement**

Append to `colab/launcher_helpers.py`:

```python
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
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests-unit/colab_test/ -v`

- [ ] **Step 5: Commit**

```bash
git add colab/launcher_helpers.py tests-unit/colab_test/test_launcher_helpers.py
git commit -m "feat(colab): add prefs.json load/save helpers"
```

---

### Task 7: Authorization header file for aria2c

**Why:** Civitai download URLs require an `Authorization: Bearer <token>` header. Putting the token on the `aria2c` CLI exposes it to `/proc`, shell history, and pip verbose error dumps. Writing it to an owner-only temp file and passing `--header=@/path/to/file` keeps the token off the command line entirely.

**Files:**
- Modify: `colab/launcher_helpers.py`
- Modify: `tests-unit/colab_test/test_launcher_helpers.py`

- [ ] **Step 1: Write failing tests**

Append to `tests-unit/colab_test/test_launcher_helpers.py`:

```python
def test_write_auth_header_creates_file_with_mode_0600(tmp_path):
    from colab.launcher_helpers import write_auth_header
    p = write_auth_header("token_abc", tmp_path)
    assert p.exists()
    assert p.read_text() == "Authorization: Bearer token_abc\n"
    # Owner-only permissions
    import stat as _stat
    mode = p.stat().st_mode & 0o777
    assert mode == 0o600


def test_write_auth_header_returns_none_for_empty_token(tmp_path):
    from colab.launcher_helpers import write_auth_header
    assert write_auth_header("", tmp_path) is None
    assert write_auth_header(None, tmp_path) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests-unit/colab_test/ -v`

- [ ] **Step 3: Implement**

Append to `colab/launcher_helpers.py`:

```python
import tempfile

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
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests-unit/colab_test/ -v`

- [ ] **Step 5: Commit**

```bash
git add colab/launcher_helpers.py tests-unit/colab_test/test_launcher_helpers.py
git commit -m "feat(colab): add mode-0600 Authorization header file writer"
```

---

### Task 8: Download command builders (`hf` + `aria2c`)

**Why:** Pure functions that return the exact argv lists for HuggingFace (`hf download`) and CivitAI (`aria2c`) downloads. Testable without subprocess execution; consumed by the notebook's Launch callback.

**Files:**
- Modify: `colab/launcher_helpers.py`
- Modify: `tests-unit/colab_test/test_launcher_helpers.py`

- [ ] **Step 1: Write failing tests**

Append to `tests-unit/colab_test/test_launcher_helpers.py`:

```python
def test_build_hf_download_cmd():
    from colab.launcher_helpers import build_hf_download_cmd
    cmd = build_hf_download_cmd(
        repo="black-forest-labs/FLUX.2-dev",
        filename="flux2-dev.safetensors",
        dest_dir="/content/ComfyUI/models/diffusion_models",
    )
    assert cmd[0] == "hf"
    assert cmd[1] == "download"
    assert "black-forest-labs/FLUX.2-dev" in cmd
    assert "flux2-dev.safetensors" in cmd
    assert "--local-dir" in cmd
    assert "/content/ComfyUI/models/diffusion_models" in cmd
    assert "--max-workers" in cmd


def test_build_aria2c_cmd_without_auth():
    from colab.launcher_helpers import build_aria2c_cmd
    cmd = build_aria2c_cmd(
        url="https://example/file.safetensors",
        dest_dir="/tmp/out",
        filename="file.safetensors",
    )
    assert cmd[0] == "aria2c"
    assert "-x" in cmd and "16" in cmd
    assert "-s" in cmd and "16" in cmd
    assert "-d" in cmd and "/tmp/out" in cmd
    assert "-o" in cmd and "file.safetensors" in cmd
    assert "https://example/file.safetensors" in cmd
    assert not any("--header" in c for c in cmd)


def test_build_aria2c_cmd_with_auth_header_file():
    from colab.launcher_helpers import build_aria2c_cmd
    cmd = build_aria2c_cmd(
        url="https://civitai.com/api/download/models/1",
        dest_dir="/tmp/out",
        filename="x.safetensors",
        auth_header_file="/tmp/auth.hdr",
    )
    assert "--header=@/tmp/auth.hdr" in cmd
    # Quiet flags to avoid leaking the token via verbose error output
    assert "--quiet=true" in cmd or any(c.startswith("--console-log-level") for c in cmd)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests-unit/colab_test/ -v`

- [ ] **Step 3: Implement**

Append to `colab/launcher_helpers.py`:

```python
__all__ += ["build_hf_download_cmd", "build_aria2c_cmd"]


def build_hf_download_cmd(
    repo: str,
    filename: str,
    dest_dir: str,
    max_workers: int = 32,
) -> List[str]:
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
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests-unit/colab_test/ -v`

- [ ] **Step 5: Commit**

```bash
git add colab/launcher_helpers.py tests-unit/colab_test/test_launcher_helpers.py
git commit -m "feat(colab): add hf + aria2c download command builders"
```

---

### Task 9: Cloudflared URL extractor

**Why:** `cloudflared tunnel --url http://127.0.0.1:8188` prints the public `https://<random>.trycloudflare.com` URL to stdout mixed with other log noise. A regex extractor run line-by-line from a background thread is the minimum needed to surface the URL to the user.

**Files:**
- Modify: `colab/launcher_helpers.py`
- Modify: `tests-unit/colab_test/test_launcher_helpers.py`

- [ ] **Step 1: Write failing tests**

Append to `tests-unit/colab_test/test_launcher_helpers.py`:

```python
def test_extract_trycloudflare_url_positive_match():
    from colab.launcher_helpers import extract_trycloudflare_url
    line = "2026-04-18T12:00:00Z INF +-----https://foo-bar-baz.trycloudflare.com-----+"
    assert extract_trycloudflare_url(line) == "https://foo-bar-baz.trycloudflare.com"


def test_extract_trycloudflare_url_returns_none_when_absent():
    from colab.launcher_helpers import extract_trycloudflare_url
    assert extract_trycloudflare_url("some unrelated log line") is None


def test_extract_trycloudflare_url_ignores_http_only():
    from colab.launcher_helpers import extract_trycloudflare_url
    assert extract_trycloudflare_url("http://not-https.trycloudflare.com") is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests-unit/colab_test/ -v`

- [ ] **Step 3: Implement**

Append to `colab/launcher_helpers.py`:

```python
__all__ += ["extract_trycloudflare_url"]


_TRYCLOUDFLARE_RX = re.compile(r"https://[-a-z0-9]+\.trycloudflare\.com")


def extract_trycloudflare_url(line: str) -> "str | None":
    m = _TRYCLOUDFLARE_RX.search(line)
    return m.group(0) if m else None
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests-unit/colab_test/ -v`

- [ ] **Step 5: Commit**

```bash
git add colab/launcher_helpers.py tests-unit/colab_test/test_launcher_helpers.py
git commit -m "feat(colab): add cloudflared URL extractor"
```

---

### Task 10: Scaffold empty Colab notebook

**Why:** Build the `.ipynb` file programmatically with `nbformat` so the JSON is well-formed and cell metadata is correct. Subsequent tasks append cell content.

**Files:**
- Create: `colab/ComfyUI_Colab.ipynb`

- [ ] **Step 1: Install nbformat in the local venv if absent**

Run: `.venv/bin/python -c "import nbformat" 2>/dev/null || .venv/bin/python -m uv pip install nbformat`

- [ ] **Step 2: Create the notebook with nine empty labelled cells**

Run (from repo root):

```bash
.venv/bin/python - <<'PY'
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
    "accelerator": "GPU",
    "colab": {"provenance": [], "gpuType": "T4"},
}
titles = [
    "# ComfyUI on Colab — Personalised Launcher",
    "## 1. Mount Google Drive (optional, for prefs only)",
    "## 2. Detect GPU and check torch compatibility",
    "## 3. Load secrets (HF_TOKEN, CIVITAI_TOKEN)",
    "## 4. Install lightweight dependencies",
    "## 5. Clone the ComfyUI repo",
    "## 6. Pipeline and LoRA selector",
    "## 7. Launch ComfyUI (install reqs, download weights, start main.py)",
    "## 8. Public URL via Cloudflared quick tunnel",
]
cells = [nbf.v4.new_markdown_cell(titles[0])]
for t in titles[1:]:
    cells.append(nbf.v4.new_markdown_cell(t))
    cells.append(nbf.v4.new_code_cell("# filled in a later task"))
nb["cells"] = cells
nbf.write(nb, "colab/ComfyUI_Colab.ipynb")
print("wrote colab/ComfyUI_Colab.ipynb")
PY
```

- [ ] **Step 3: Verify the notebook is valid JSON and parses with nbformat**

Run:

```bash
.venv/bin/python -c "
import nbformat
nb = nbformat.read('colab/ComfyUI_Colab.ipynb', as_version=4)
assert len(nb.cells) == 17  # 1 title + 8 sections * 2 (md+code) = 17
assert nb.metadata['accelerator'] == 'GPU'
print('OK', len(nb.cells), 'cells')
"
```

Expected: `OK 17 cells`.

- [ ] **Step 4: Commit**

```bash
git add colab/ComfyUI_Colab.ipynb
git commit -m "feat(colab): scaffold ComfyUI_Colab notebook with eight sections"
```

---

### Task 11: Fill notebook setup cells (1 → 5)

**Why:** Cells 1–5 are thin orchestration cells that call the helpers built in Tasks 1–9. They mount Drive, detect GPU, load secrets, install light deps, and clone the repo. Each cell is short enough to append directly via nbformat.

**Files:**
- Modify: `colab/ComfyUI_Colab.ipynb`

- [ ] **Step 1: Write a Python script that rewrites the five setup cells**

Run (from repo root):

```bash
.venv/bin/python - <<'PY'
import nbformat as nbf

path = "colab/ComfyUI_Colab.ipynb"
nb = nbf.read(path, as_version=4)

# Cell indexes (after section headers):
# 0 title md, 1 sec1 md, 2 sec1 code, 3 sec2 md, 4 sec2 code,
# 5 sec3 md, 6 sec3 code, 7 sec4 md, 8 sec4 code, 9 sec5 md, 10 sec5 code
cells_by_section = {1: 2, 2: 4, 3: 6, 4: 8, 5: 10}

cell_sources = {
    1: '''\
import os
try:
    from google.colab import drive
    drive.mount("/content/drive", force_remount=False)
    PREFS_PATH = "/content/drive/MyDrive/.comfyui-colab/prefs.json"
    print("Drive mounted. Prefs path:", PREFS_PATH)
except Exception as e:
    PREFS_PATH = "/content/.comfyui-colab-prefs.json"
    print(f"Drive not available ({type(e).__name__}); prefs will live at {PREFS_PATH}")
''',
    2: '''\
import subprocess, sys

# Use stdlib only — helpers not yet on sys.path at this point in the notebook.
def _cc():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"], text=True
        ).strip().splitlines()[0]
        name, cc = [s.strip() for s in out.split(",", 1)]
    except Exception:
        return "CPU", 0
    try:
        maj, mn = cc.split(".")
        return name, int(maj) * 10 + int(mn)
    except Exception:
        return name, 0

GPU_NAME, GPU_CC = _cc()
print(f"GPU: {GPU_NAME}  sm_{GPU_CC if GPU_CC else 'none'}")

import torch
ARCH_LIST = list(torch.cuda.get_arch_list()) if torch.cuda.is_available() else []
print(f"torch {torch.__version__} archs: {ARCH_LIST}")

NEEDS_REINSTALL = GPU_CC != 0 and f"sm_{GPU_CC}" not in ARCH_LIST
if NEEDS_REINSTALL:
    print(f"WARNING: sm_{GPU_CC} not in preinstalled torch; reinstalling cu128…")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "--upgrade",
        "torch==2.11.0", "torchvision==0.26.0", "torchaudio==2.11.0",
        "--index-url", "https://download.pytorch.org/whl/cu128",
    ])
    print("\\nREINSTALLED. You MUST choose Runtime → Restart session before running the next cell.")
else:
    print("Preinstalled torch is fine; no reinstall needed.")
''',
    3: '''\
import os, getpass, sys

def _load(name: str):
    mod = sys.modules.get("google.colab")
    if mod is None:
        try:
            import google.colab as mod  # type: ignore
        except ImportError:
            mod = None
    if mod is not None and hasattr(mod, "userdata"):
        try:
            val = mod.userdata.get(name)
            if val:
                return val
        except Exception:
            pass
    try:
        val = getpass.getpass(f"{name} (hidden, or press Enter to skip): ")
    except Exception:
        return None
    return val or None

HF_TOKEN = _load("HF_TOKEN")
CIVITAI_TOKEN = _load("CIVITAI_TOKEN")
if HF_TOKEN:
    os.environ["HF_TOKEN"] = os.environ["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN
if CIVITAI_TOKEN:
    os.environ["CIVITAI_TOKEN"] = CIVITAI_TOKEN
print(f"HF_TOKEN: {'set' if HF_TOKEN else 'missing'}   CIVITAI_TOKEN: {'set' if CIVITAI_TOKEN else 'missing'}")
del HF_TOKEN, CIVITAI_TOKEN  # drop from IPython _oh history
''',
    4: '''\
!apt-get install -y -qq aria2 >/dev/null
!pip install -q --upgrade ipywidgets huggingface_hub
!pip install -q --upgrade "huggingface_hub[cli]"  # provides the `hf` CLI
print("Deps installed.")
''',
    5: '''\
import os, subprocess
REPO_URL = os.environ.get("COMFYUI_COLAB_REPO_URL", "https://github.com/samsam27/ComfyUI.git")
REPO_DIR = "/content/ComfyUI"
if not os.path.isdir(REPO_DIR):
    subprocess.check_call(["git", "clone", "--depth", "1", REPO_URL, REPO_DIR])
else:
    subprocess.check_call(["git", "-C", REPO_DIR, "pull", "--ff-only"])
import sys
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)
print("Repo ready at", REPO_DIR)
''',
}

for sec, idx in cells_by_section.items():
    nb.cells[idx]["source"] = cell_sources[sec]
    nb.cells[idx]["outputs"] = []
    nb.cells[idx]["execution_count"] = None

nbf.write(nb, path)
print("updated cells 1..5")
PY
```

- [ ] **Step 2: Verify the notebook still parses**

Run:

```bash
.venv/bin/python -c "
import nbformat
nb = nbformat.read('colab/ComfyUI_Colab.ipynb', as_version=4)
codes = [c for c in nb.cells if c.cell_type == 'code']
assert all(len(c.source) > 0 for c in codes[:5]), 'setup cells should be non-empty'
print('OK — setup cells filled')
"
```

- [ ] **Step 3: Commit**

```bash
git add colab/ComfyUI_Colab.ipynb
git commit -m "feat(colab): fill setup cells (Drive, GPU, secrets, deps, clone)"
```

---

### Task 12: Fill notebook selector UI cell (section 6)

**Why:** The single most user-facing cell. It reads `scripts/model_registry.json` from the cloned repo, builds a two-pane ipywidgets layout (pipeline checkboxes on the left; lora SelectMultiple on the right, repopulated when selection changes), restores last choices from `PREFS_PATH`, and exposes a module-global `STATE` dict that the Launch cell reads.

**Files:**
- Modify: `colab/ComfyUI_Colab.ipynb`

- [ ] **Step 1: Rewrite section 6 cell**

Run:

```bash
.venv/bin/python - <<'PY'
import nbformat as nbf

src = '''\
import json, os
from pathlib import Path
import ipywidgets as W
from IPython.display import display

from colab.launcher_helpers import load_registry, list_pipelines, pipeline_loras, load_prefs

REGISTRY_PATH = "/content/ComfyUI/scripts/model_registry.json"
registry = load_registry(REGISTRY_PATH)
pipelines = list_pipelines(registry)
prefs = load_prefs(PREFS_PATH)
selected_prev = set(prefs.get("pipelines", []))
lora_prev = {k: set(v) for k, v in prefs.get("loras", {}).items()}

STATE = {"pipelines": set(), "loras": {}, "prefs_path": PREFS_PATH}

pipeline_boxes = []
lora_widgets = {}

for p in pipelines:
    cb = W.Checkbox(
        value=(p["key"] in selected_prev),
        description=f"{p['display_name']}  ({p['total_size_gb']} GB, {p['model_count']} models, {p['lora_count']} LoRAs)",
        indent=False,
        layout=W.Layout(width="100%"),
    )
    pipeline_boxes.append((p["key"], cb))

def _update_lora_pane(*_):
    STATE["pipelines"] = {k for k, cb in pipeline_boxes if cb.value}
    children = []
    STATE["loras"] = {}
    for key in sorted(STATE["pipelines"]):
        loras = pipeline_loras(registry, key)
        if not loras:
            continue
        options = [(f"{l['display_name']} ({l.get('size_gb', 0):.2f} GB)", l["filename"]) for l in loras]
        preselect = [fn for _disp, fn in options if fn in lora_prev.get(key, set())]
        sm = W.SelectMultiple(
            options=options, value=tuple(preselect),
            description=key, rows=min(6, len(options)),
            layout=W.Layout(width="100%"),
        )
        lora_widgets[key] = sm
        STATE["loras"][key] = set(preselect)
        def _on_change(change, _key=key):
            STATE["loras"][_key] = set(change["new"])
        sm.observe(_on_change, names="value")
        children.append(sm)
    lora_pane.children = children

for _k, cb in pipeline_boxes:
    cb.observe(_update_lora_pane, names="value")

lora_pane = W.VBox([], layout=W.Layout(width="50%"))
left = W.VBox([W.HTML("<b>Pipelines</b>")] + [cb for _k, cb in pipeline_boxes],
              layout=W.Layout(width="50%"))
right = W.VBox([W.HTML("<b>LoRAs (per selected pipeline)</b>"), lora_pane])

launch_btn = W.Button(description="Launch ComfyUI", button_style="success", icon="rocket")
status = W.HTML("<i>Pick at least one pipeline, then click Launch.</i>")

def _on_launch(_btn):
    if not STATE["pipelines"]:
        status.value = "<span style='color:#a00'>Select at least one pipeline first.</span>"
        return
    status.value = "<b>Launching…</b> scroll to section 7 for progress."
    from colab.launcher_helpers import save_prefs
    save_prefs(PREFS_PATH, {
        "pipelines": sorted(STATE["pipelines"]),
        "loras": {k: sorted(v) for k, v in STATE["loras"].items()},
    })
    launch_btn.disabled = True
    STATE["launch_triggered"] = True

launch_btn.on_click(_on_launch)
_update_lora_pane()  # initial render
display(W.VBox([W.HBox([left, right]), launch_btn, status]))
'''

path = "colab/ComfyUI_Colab.ipynb"
nb = nbf.read(path, as_version=4)
# Section 6 code cell index: after title + 6*(md,code) structure: md at 11, code at 12
nb.cells[12]["source"] = src
nb.cells[12]["outputs"] = []
nb.cells[12]["execution_count"] = None
nbf.write(nb, path)
print("updated cell 12 (selector UI)")
PY
```

- [ ] **Step 2: Verify the notebook parses**

Run:

```bash
.venv/bin/python -c "
import nbformat
nb = nbformat.read('colab/ComfyUI_Colab.ipynb', as_version=4)
assert 'SelectMultiple' in nb.cells[12].source
assert 'STATE' in nb.cells[12].source
print('OK')
"
```

- [ ] **Step 3: Commit**

```bash
git add colab/ComfyUI_Colab.ipynb
git commit -m "feat(colab): add ipywidgets pipeline/LoRA selector with prefs round-trip"
```

---

### Task 13: Fill notebook launch + tunnel cells (sections 7 & 8)

**Why:** Section 7 is the big one — it consumes `STATE`, installs `requirements.txt`, downloads weights for the selected pipelines/LoRAs using the Task 8 builders, installs each `custom_nodes/*/requirements.txt` (best-effort), and starts `main.py`. Section 8 launches cloudflared in a daemon thread and surfaces the `*.trycloudflare.com` URL as soon as it appears.

**Files:**
- Modify: `colab/ComfyUI_Colab.ipynb`

- [ ] **Step 1: Rewrite cells 14 (section 7) and 16 (section 8)**

Run:

```bash
.venv/bin/python - <<'PY'
import nbformat as nbf

launch_src = '''\
import os, subprocess, sys, tempfile
from pathlib import Path

from colab.launcher_helpers import (
    pipeline_models, pipeline_loras,
    build_hf_download_cmd, build_aria2c_cmd, write_auth_header,
)

if not STATE.get("launch_triggered"):
    print("No pipelines launched yet. Go back to section 6, pick pipelines, and click **Launch**.")
    print("Then come back and run this cell (and section 8) to start the downloads and tunnel.")
    raise SystemExit(0)
MODELS_ROOT = Path("/content/ComfyUI/models")

def _run(cmd, label):
    print(f"  → {label}")
    rc = subprocess.call(cmd)
    if rc != 0:
        print(f"    [warn] exit {rc} for: {' '.join(cmd[:4])}…")

# 1. Install ComfyUI core requirements
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-r",
                       "/content/ComfyUI/requirements.txt"])

# 2. Install custom-node requirements best-effort
for req in Path("/content/ComfyUI/custom_nodes").glob("*/requirements.txt"):
    subprocess.call([sys.executable, "-m", "pip", "install", "-q", "-r", str(req)])

# 3. Download models for each selected pipeline
civitai_hdr = None
if os.environ.get("CIVITAI_TOKEN"):
    civitai_hdr = write_auth_header(os.environ["CIVITAI_TOKEN"], tempfile.mkdtemp())

for pk in sorted(STATE["pipelines"]):
    for m in pipeline_models(registry, pk):
        dest = MODELS_ROOT / m["dest_subdir"]
        dest.mkdir(parents=True, exist_ok=True)
        if (dest / m["filename"]).exists():
            print(f"  ✓ {m['filename']} already present")
            continue
        if m["source"] == "hf":
            _run(build_hf_download_cmd(m["hf_repo"], m["hf_file"], str(dest)),
                 f"hf: {m['filename']}")
            # hf may preserve path depth; ensure the final filename lands flat
            got = dest / Path(m["hf_file"]).name
            if got.exists() and got.name != m["filename"]:
                got.rename(dest / m["filename"])
        elif m["source"] == "civitai":
            _run(build_aria2c_cmd(m["civitai_url"], str(dest), m["filename"], str(civitai_hdr) if civitai_hdr else None),
                 f"civitai: {m['filename']}")
    for lora_fn in sorted(STATE["loras"].get(pk, [])):
        spec = next((l for l in pipeline_loras(registry, pk) if l["filename"] == lora_fn), None)
        if not spec:
            continue
        dest = MODELS_ROOT / "loras"
        dest.mkdir(parents=True, exist_ok=True)
        if (dest / spec["filename"]).exists():
            continue
        if spec["source"] == "hf":
            _run(build_hf_download_cmd(spec["hf_repo"], spec["hf_file"], str(dest)),
                 f"hf lora: {spec['filename']}")
        elif spec["source"] == "civitai":
            _run(build_aria2c_cmd(spec["civitai_url"], str(dest), spec["filename"], str(civitai_hdr) if civitai_hdr else None),
                 f"civitai lora: {spec['filename']}")

# 4. Start ComfyUI in the background
LOG = "/content/comfyui.log"
open(LOG, "w").close()
os.chdir("/content/ComfyUI")
proc = subprocess.Popen(
    [sys.executable, "main.py", "--listen", "0.0.0.0", "--port", "8188",
     "--enable-cors-header", "--dont-print-server"],
    stdout=open(LOG, "a"), stderr=subprocess.STDOUT,
)
STATE["comfy_pid"] = proc.pid
print(f"ComfyUI starting (pid {proc.pid}); tail log with: !tail -f /content/comfyui.log")
'''

tunnel_src = '''\
import os, re, socket, subprocess, threading, time, urllib.request
from colab.launcher_helpers import extract_trycloudflare_url

if not os.path.exists("/usr/local/bin/cloudflared"):
    urllib.request.urlretrieve(
        "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64",
        "/usr/local/bin/cloudflared")
    os.chmod("/usr/local/bin/cloudflared", 0o755)

PORT = 8188
PUBLIC_URL = {"url": None}

def _tunnel():
    # Wait for ComfyUI to open the port
    for _ in range(180):  # up to 90s
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", PORT)) == 0:
                break
        time.sleep(0.5)
    p = subprocess.Popen(
        ["cloudflared", "tunnel", "--no-autoupdate", "--url", f"http://127.0.0.1:{PORT}"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
    )
    for line in p.stdout:
        u = extract_trycloudflare_url(line)
        if u and not PUBLIC_URL["url"]:
            PUBLIC_URL["url"] = u
            print("\\n" + "=" * 60)
            print(f"  ComfyUI public URL: {u}")
            print("=" * 60 + "\\n")
    p.wait()

threading.Thread(target=_tunnel, daemon=True).start()
print("Tunnel starting… URL will print here once ComfyUI is ready (typically 30-90 s).")
'''

path = "colab/ComfyUI_Colab.ipynb"
nb = nbf.read(path, as_version=4)
# Section 7 code cell index: md 13, code 14
# Section 8 code cell index: md 15, code 16
nb.cells[14]["source"] = launch_src
nb.cells[14]["outputs"] = []
nb.cells[14]["execution_count"] = None
nb.cells[16]["source"] = tunnel_src
nb.cells[16]["outputs"] = []
nb.cells[16]["execution_count"] = None
nbf.write(nb, path)
print("updated cells 14 and 16")
PY
```

- [ ] **Step 2: Verify the notebook parses**

Run:

```bash
.venv/bin/python -c "
import nbformat
nb = nbformat.read('colab/ComfyUI_Colab.ipynb', as_version=4)
assert 'cloudflared' in nb.cells[16].source
assert 'build_hf_download_cmd' in nb.cells[14].source
assert 'build_aria2c_cmd' in nb.cells[14].source
print('OK')
"
```

- [ ] **Step 3: Commit**

```bash
git add colab/ComfyUI_Colab.ipynb
git commit -m "feat(colab): add Launch callback and cloudflared tunnel cells"
```

---

### Task 14: Add `colab/README.md` with Open-in-Colab link

**Why:** Sam (and future collaborators) need a one-click entry point. The README documents the prerequisites (Pro+, Drive permission, HF_TOKEN/CIVITAI_TOKEN in userdata), shows the Open-in-Colab badge, and explains the cell flow.

**Files:**
- Create: `colab/README.md`

- [ ] **Step 1: Write the README**

Write `colab/README.md`:

```markdown
# ComfyUI on Colab — Personalised Launcher

One-click launch of this repo's ComfyUI setup on Google Colab, with registry-driven
pipeline selection, on-demand weight downloads, and a public `cloudflared` URL.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/samsam27/ComfyUI/blob/master/colab/ComfyUI_Colab.ipynb)

Update the badge link if you fork this repo: replace `samsam27/ComfyUI` with `<your-gh-user>/ComfyUI`.

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
```

- [ ] **Step 2: Verify the file renders**

Run: `head -5 colab/README.md`
Expected: first line is `# ComfyUI on Colab — Personalised Launcher`.

- [ ] **Step 3: Commit**

```bash
git add colab/README.md
git commit -m "docs(colab): add README with Open-in-Colab badge and caveats"
```

---

## Post-plan verification

After all tasks complete:

- [ ] Run the full test suite: `.venv/bin/python -m pytest tests-unit/colab_test/ -v`. All tests must pass.
- [ ] Run `nbformat validate`: `.venv/bin/python -c "import nbformat; nbformat.validate(nbformat.read('colab/ComfyUI_Colab.ipynb', as_version=4))"`. No output = valid.
- [ ] Smoke-test the notebook on an actual Colab runtime: open the Open-in-Colab link, run cells top-to-bottom with Z-Image Turbo 6B selected (fits on the smallest GPU), confirm a `trycloudflare.com` URL loads the ComfyUI web UI.

The Colab smoke test cannot be automated from the workstation — this is the manual verification step.
