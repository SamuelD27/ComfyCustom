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
