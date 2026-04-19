from colab import launcher_helpers
from unittest.mock import patch
import subprocess


def test_version_is_set():
    assert launcher_helpers.__version__ == "0.1.0"


def test_compute_cap_from_smi_parses_standard_output():
    fake = "8.0\n"
    with patch("subprocess.check_output", return_value=fake):
        assert launcher_helpers.compute_cap_from_smi() == 80


def test_compute_cap_from_smi_handles_multiple_gpus_picks_first():
    fake = "9.0\n7.5\n"
    with patch("subprocess.check_output", return_value=fake):
        assert launcher_helpers.compute_cap_from_smi() == 90


def test_compute_cap_from_smi_returns_zero_on_failure():
    with patch(
        "subprocess.check_output",
        side_effect=subprocess.CalledProcessError(1, "nvidia-smi"),
    ):
        assert launcher_helpers.compute_cap_from_smi() == 0


def test_compute_cap_from_smi_returns_zero_when_nvidia_smi_missing():
    with patch("subprocess.check_output", side_effect=FileNotFoundError):
        assert launcher_helpers.compute_cap_from_smi() == 0


def test_compute_cap_from_name_maps_known_gpus():
    assert launcher_helpers.compute_cap_from_name("NVIDIA A100-SXM4-40GB") == 80
    assert launcher_helpers.compute_cap_from_name("NVIDIA H100 80GB HBM3") == 90
    assert launcher_helpers.compute_cap_from_name("NVIDIA H200") == 90
    assert launcher_helpers.compute_cap_from_name("NVIDIA RTX PRO 6000 Blackwell") == 120
    assert launcher_helpers.compute_cap_from_name("NVIDIA L4") == 89
    assert launcher_helpers.compute_cap_from_name("NVIDIA L40S") == 89
    assert launcher_helpers.compute_cap_from_name("Tesla T4") == 75
    assert launcher_helpers.compute_cap_from_name("Some Unknown GPU") == 0


def test_torch_action_keep_when_cc_in_archlist():
    decision = launcher_helpers.torch_action(cc=80, arch_list=["sm_80", "sm_90", "sm_120"])
    assert decision.action == "keep"
    assert decision.reason


def test_torch_action_keep_when_cc_is_zero_cpu_only_runtime():
    decision = launcher_helpers.torch_action(cc=0, arch_list=["sm_80"])
    assert decision.action == "keep"


def test_torch_action_reinstall_when_cc_missing():
    decision = launcher_helpers.torch_action(cc=120, arch_list=["sm_80", "sm_90"])
    assert decision.action == "reinstall"
    assert "cu128" in decision.reason


def test_torch_action_reinstall_returns_pip_args():
    decision = launcher_helpers.torch_action(cc=120, arch_list=["sm_80"])
    assert decision.pip_args[0].startswith("torch==")
    assert decision.index_url.endswith("/cu128")


import os
import sys
from unittest.mock import patch


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
        assert launcher_helpers.load_secret("HF_TOKEN") == "hf_fake_token_abc"


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
        assert launcher_helpers.load_secret("HF_TOKEN", interactive=False) is None


def test_load_secret_falls_back_to_getpass_when_no_colab(monkeypatch):
    # Simulate non-Colab environment: google.colab not importable
    monkeypatch.setitem(__import__("sys").modules, "google.colab", None)
    with patch("getpass.getpass", return_value="hf_from_prompt"):
        assert launcher_helpers.load_secret("HF_TOKEN", interactive=True) == "hf_from_prompt"


def test_apply_hf_env_sets_both_variables(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    launcher_helpers.apply_hf_env("hf_xyz")
    assert os.environ["HF_TOKEN"] == "hf_xyz"
    assert os.environ["HUGGING_FACE_HUB_TOKEN"] == "hf_xyz"


def test_apply_hf_env_noop_on_none(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    launcher_helpers.apply_hf_env(None)
    assert "HF_TOKEN" not in os.environ


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
