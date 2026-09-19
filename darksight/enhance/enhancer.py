"""Inference wrapper around Zero-DCE: uint8 RGB in, uint8 RGB out."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ..constants import DEFAULT_ZERODCE_WEIGHTS
from ..device import select_device
from .zerodce_model import ZeroDCENet


class ZeroDCEEnhancer:
    def __init__(self, weights: str | Path = DEFAULT_ZERODCE_WEIGHTS, device: str | None = None,
                 half: bool = False):
        self.device = select_device(device)
        self.half = half and self.device.startswith("cuda")
        self.model = ZeroDCENet()
        state = torch.load(str(weights), map_location="cpu", weights_only=True)
        state = {k.removeprefix("module."): v for k, v in state.items()}
        self.model.load_state_dict(state)
        self.model.to(self.device).eval()
        if self.half:
            self.model.half()

    @torch.inference_mode()
    def enhance_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """x: Nx3xHxW float in [0, 1] -> same shape, clamped to [0, 1]."""
        x = x.to(self.device, dtype=torch.float16 if self.half else torch.float32)
        _, out, _ = self.model(x)
        return out.clamp_(0.0, 1.0).float()

    def enhance(self, img: np.ndarray) -> np.ndarray:
        """img: HxWx3 uint8 RGB -> HxWx3 uint8 RGB of the same size."""
        if img.ndim != 3 or img.shape[2] != 3 or img.dtype != np.uint8:
            raise ValueError(f"expected HxWx3 uint8 RGB, got {img.shape} {img.dtype}")
        x = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().div_(255.0)
        out = self.enhance_tensor(x)[0].permute(1, 2, 0).cpu().numpy()
        return (out * 255.0 + 0.5).astype(np.uint8)

    def __call__(self, img: np.ndarray) -> np.ndarray:
        return self.enhance(img)
