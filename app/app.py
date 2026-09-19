#!/usr/bin/env python
"""DarkSight demo: upload a dark image -> Zero-DCE enhancement -> YOLOv8 detection, raw vs enhanced.

    python app/app.py                 # http://127.0.0.1:7860
    python app/app.py --share         # temporary public link (Colab)

Uses weights/det_raw.pt and weights/det_enh.pt when present; otherwise falls back to the
COCO-pretrained yolov8s.pt with classes remapped to ExDark (clearly labelled in the UI).
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import argparse  # noqa: E402

import gradio as gr  # noqa: E402
import pandas as pd  # noqa: E402

from darksight.detect.detector import Detector  # noqa: E402
from darksight.device import select_device  # noqa: E402
from darksight.enhance.enhancer import ZeroDCEEnhancer  # noqa: E402
from darksight.pipeline import DarkSightPipeline  # noqa: E402
from darksight.visualize import draw_boxes  # noqa: E402

WEIGHTS = ROOT / "weights"
EXAMPLES = ROOT / "app" / "examples"
_PIPELINE = None
_MODEL_NOTE = ""


def get_pipeline(device=None) -> DarkSightPipeline:
    """Load all models once (first request) and reuse them."""
    global _PIPELINE, _MODEL_NOTE
    if _PIPELINE is None:
        device = select_device(device)
        raw_w, enh_w = WEIGHTS / "det_raw.pt", WEIGHTS / "det_enh.pt"
        if raw_w.exists():
            det_raw = Detector(raw_w, device=device)
            det_enh = Detector(enh_w, device=device) if enh_w.exists() else det_raw
            _MODEL_NOTE = (f"Detectors: `{raw_w.name}` (raw) / `{(enh_w if enh_w.exists() else raw_w).name}` "
                           f"(enhanced) on `{device}`")
        else:
            det_raw = det_enh = Detector("yolov8s.pt", device=device, coco_remap=True)
            _MODEL_NOTE = (f"⚠️ Fine-tuned weights not found — using COCO `yolov8s.pt` zero-shot "
                           f"(classes mapped to ExDark) on `{device}`")
        _PIPELINE = DarkSightPipeline(ZeroDCEEnhancer(device=device), det_raw, det_enh)
    return _PIPELINE


def run(image, mode, conf, use_gate, tau):
    if image is None:
        raise gr.Error("Upload an image first.")
    pipe = get_pipeline()
    res = pipe.run(image, mode={"Raw only": "raw", "Enhanced only": "enhanced"}.get(mode, "both"),
                   conf=conf, gate_tau=tau if use_gate else None)

    raw_vis = enh_vis = None
    rows = []
    if res.raw_detections is not None:
        d = res.raw_detections
        raw_vis = draw_boxes(res.input_image, d.xyxy, d.cls, d.conf)
        rows += [{"pipeline": "raw", **r} for r in d.as_rows()]
    if res.enhanced_detections is not None:
        d = res.enhanced_detections
        enh_vis = draw_boxes(res.enhanced_image, d.xyxy, d.cls, d.conf)
        rows += [{"pipeline": "enhanced", **r} for r in d.as_rows()]

    t = res.timings_ms
    lines = [
        _MODEL_NOTE,
        f"Mean luminance **{res.luminance:.3f}**"
        + (f" — enhancement {'applied' if res.enhancement_applied else 'skipped (gate)'}" if res.enhanced_image is not None else ""),
        " · ".join(f"{k}: {v:.0f} ms" for k, v in t.items()),
    ]
    if res.raw_detections is not None and res.enhanced_detections is not None:
        lines.append(f"Detections — raw: **{len(res.raw_detections)}**, enhanced: **{len(res.enhanced_detections)}**")
    table = pd.DataFrame(rows, columns=["pipeline", "class", "conf", "box"])
    return raw_vis, res.enhanced_image, enh_vis, table, "\n\n".join(lines)


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="DarkSight") as demo:
        gr.Markdown("# DarkSight\nLow-light object detection: **Zero-DCE** enhancement → **YOLOv8** detection. "
                    "Compare detections on the raw image against the enhanced pipeline.")
        with gr.Row():
            with gr.Column(scale=1):
                inp = gr.Image(type="numpy", label="Input image", height=320)
                mode = gr.Radio(["Side-by-side", "Raw only", "Enhanced only"], value="Side-by-side", label="Mode")
                conf = gr.Slider(0.05, 0.9, value=0.25, step=0.05, label="Confidence threshold")
                with gr.Row():
                    use_gate = gr.Checkbox(value=False, label="Luminance gate")
                    tau = gr.Slider(0.05, 0.8, value=0.35, step=0.01, label="Gate τ (enhance only if darker)")
                btn = gr.Button("Detect", variant="primary")
                examples = sorted(str(p) for p in EXAMPLES.glob("*.jpg"))
                if examples:
                    gr.Examples(examples, inputs=inp, label="ExDark test examples")
            with gr.Column(scale=2):
                info = gr.Markdown()
                with gr.Row():
                    out_raw = gr.Image(label="Raw + detections")
                    out_enh_det = gr.Image(label="Enhanced + detections")
                out_enh = gr.Image(label="Zero-DCE enhanced image")
                table = gr.Dataframe(label="Detections", wrap=True)
        btn.click(run, [inp, mode, conf, use_gate, tau], [out_raw, out_enh, out_enh_det, table, info])
    return demo


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    get_pipeline(args.device)  # load before serving so the first click is fast
    build_ui().launch(share=args.share, server_port=args.port)
