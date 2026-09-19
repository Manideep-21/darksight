import os

import torch


def select_device(preferred: str | None = None) -> str:
    """cuda -> mps -> cpu, unless explicitly given (e.g. 'cpu', 'cuda:0', 'mps')."""
    if preferred and preferred != "auto":
        return preferred
    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        return "mps"
    return "cpu"
