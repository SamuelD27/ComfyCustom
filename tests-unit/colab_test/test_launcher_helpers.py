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
