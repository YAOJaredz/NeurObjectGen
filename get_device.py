import torch

def get_device() -> str:
    """Return the best available device (GPU if available, else CPU)."""
    return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"