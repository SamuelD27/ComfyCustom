from colab import launcher_helpers


def test_version_is_set():
    assert launcher_helpers.__version__ == "0.1.0"
