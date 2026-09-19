"""End-to-end smoke test. Needs yolov8n.pt (auto-downloaded by Ultralytics on first run)."""

import numpy as np
import pytest

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def pipeline():
    from darksight.detect.detector import Detector
    from darksight.enhance import ZeroDCEEnhancer
    from darksight.constants import PROJECT_ROOT
    from darksight.pipeline import DarkSightPipeline

    local = PROJECT_ROOT / "weights" / "yolov8n.pt"
    try:
        det = Detector(local if local.exists() else "yolov8n.pt", device="cpu", coco_remap=True)
    except Exception as e:  # offline
        pytest.skip(f"YOLO weights unavailable: {e}")
    return DarkSightPipeline(ZeroDCEEnhancer(device="cpu"), det, max_side=320)


def test_both_modes(pipeline):
    img = (np.random.default_rng(0).random((480, 640, 3)) * 50).astype(np.uint8)
    res = pipeline.run(img, mode="both", conf=0.25)
    assert res.input_image.shape == (240, 320, 3)  # resized to max_side
    assert res.enhanced_image.shape == res.input_image.shape
    assert res.enhancement_applied
    assert set(res.timings_ms) == {"detect_raw", "enhance", "detect_enhanced"}
    for d in (res.raw_detections, res.enhanced_detections):
        assert d.xyxy.shape[1:] == (4,) and ((d.cls >= 0) & (d.cls < 12)).all()


def test_gate_skips_bright_images(pipeline):
    img = np.full((100, 100, 3), 220, dtype=np.uint8)
    res = pipeline.run(img, mode="enhanced", gate_tau=0.35)
    assert not res.enhancement_applied and np.array_equal(res.enhanced_image, res.input_image)
