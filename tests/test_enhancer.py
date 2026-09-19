import numpy as np
import pytest
import torch

from darksight.enhance import ZeroDCEEnhancer, mean_luminance, should_enhance
from darksight.enhance.classical import clahe, gamma
from darksight.enhance.zerodce_model import ZeroDCENet


@pytest.fixture(scope="module")
def enhancer():
    return ZeroDCEEnhancer(device="cpu")


def test_official_weights_load(enhancer):
    assert sum(p.numel() for p in enhancer.model.parameters()) == 79416


def test_shape_dtype_and_brightening(enhancer):
    rng = np.random.default_rng(0)
    img = (rng.random((61, 97, 3)) * 40).astype(np.uint8)  # odd sizes on purpose
    out = enhancer.enhance(img)
    assert out.shape == img.shape and out.dtype == np.uint8
    assert mean_luminance(out) > mean_luminance(img) + 0.1


def test_curve_formula_matches_paper():
    """With zero conv weights and bias -> tanh(b) curves; check LE(x) = x + a*x*(1-x) iterated 8x (a = -r)."""
    net = ZeroDCENet()
    for m in net.modules():
        if isinstance(m, torch.nn.Conv2d):
            torch.nn.init.zeros_(m.weight); torch.nn.init.zeros_(m.bias)
    net.e_conv7.bias.data.fill_(-0.5)  # r = tanh(-0.5) -> a = 0.462
    x = torch.full((1, 3, 4, 4), 0.2)
    _, y, _ = net(x)
    a, v = -np.tanh(-0.5), 0.2
    for _ in range(8):
        v = v + a * v * (1 - v)
    assert torch.allclose(y, torch.full_like(y, v), atol=1e-6)


def test_rejects_wrong_input(enhancer):
    with pytest.raises(ValueError):
        enhancer.enhance(np.zeros((10, 10), dtype=np.uint8))


def test_gating():
    dark = np.full((8, 8, 3), 20, dtype=np.uint8)
    bright = np.full((8, 8, 3), 200, dtype=np.uint8)
    assert should_enhance(dark, 0.35) and not should_enhance(bright, 0.35)
    assert mean_luminance(bright) == pytest.approx(200 / 255, abs=1e-4)


def test_classical_baselines_preserve_shape():
    img = (np.random.default_rng(1).random((32, 48, 3)) * 60).astype(np.uint8)
    for fn in (clahe, gamma):
        out = fn(img)
        assert out.shape == img.shape and out.dtype == np.uint8
    assert gamma(img).mean() > img.mean()
