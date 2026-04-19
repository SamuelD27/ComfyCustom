"""Pure-Python helpers for the Colab ComfyUI launcher.

All functions here must be importable in an environment with only the
Python stdlib — heavy imports (huggingface_hub, ipywidgets, nbformat,
google.colab) go inside functions so tests run without them installed.
"""

__all__ = ["__version__"]

__version__ = "0.1.0"
