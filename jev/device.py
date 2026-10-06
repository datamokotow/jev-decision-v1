import os

import torch


def pick_device():
    """JEV_DEVICE env override, else cuda > mps (Apple GPU) > cpu."""
    forced = os.environ.get("JEV_DEVICE")
    if forced:
        return forced
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
