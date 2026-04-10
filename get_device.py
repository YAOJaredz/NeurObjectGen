import torch

def get_device() -> torch.device:
    """Return the best available device (GPU if available, else CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")