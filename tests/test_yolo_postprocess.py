"""Tests for the YOLOv8 postprocess functions.

Builds synthetic raw ONNX outputs of shape (1, 4+nc, A) and asserts the
decode + NMS + letterbox-undo + crop pipeline produces the right boxes.
"""
from __future__ import annotations

import os

os.environ.setdefault("FINWAVE_API_KEY", "test-key")

import numpy as np
import pytest
from PIL import Image

from finwave_inference_server.postprocess._registry import PostprocessCtx, get
from finwave_inference_server.postprocess.yolo import (
    _decode_v8,
    _nms,
    _undo_letterbox,
)
from finwave_inference_server.schemas import (
    ArtifactFile,
    ArtifactSpec,
    FieldSpec,
    FinwaveModelCard,
    InferenceConfig,
    InputSpec,
    OutputSpec,
)


def _make_card(image_size: int = 640) -> FinwaveModelCard:
    """Minimal detector ModelCard for tests."""
    return FinwaveModelCard(
        spec_version="1.0.0",
        job_id="test-job",
        model_name="test-detector",
        node_type="Detector",
        input=InputSpec(contract="RawImage", image_size=image_size, channels=3, normalization="none"),
        output=OutputSpec(
            contract="DetectionSet",
            fields=[
                FieldSpec(name="ProportionBoxes", type="list", source="postprocess:yolo_boxes_proportional"),
                FieldSpec(name="AbsoluteBoxes",   type="list", source="postprocess:yolo_boxes_absolute"),
                FieldSpec(name="ExtractedImages", type="list", source="postprocess:yolo_crops_base64"),
                FieldSpec(name="Confidences",     type="list", source="postprocess:yolo_confidences"),
            ],
            produces_embedding=False,
            class_dict={"0": "fin"},
        ),
        inference_config=InferenceConfig(conf_threshold=0.25, iou_threshold=0.5),
        artifact=ArtifactSpec(
            format="onnx-bundle",
            entrypoint="encoder.onnx",
            files=[ArtifactFile(name="encoder.onnx", url="https://example/x", sha256="0" * 64)],
        ),
    )


def _raw_v8(boxes_cxcywh: list[tuple[float, float, float, float]], scores: list[float]) -> np.ndarray:
    """Build a YOLOv8-shaped output: (1, 4+nc, A) with nc=1.

    Each anchor's first 4 channels are (cx, cy, w, h), 5th is the class score.
    Pads to a minimum of 8 anchors so tests look like real YOLO outputs.
    """
    n = max(len(boxes_cxcywh), 8)
    raw = np.zeros((1, 5, n), dtype=np.float32)
    for i, ((cx, cy, w, h), s) in enumerate(zip(boxes_cxcywh, scores)):
        raw[0, 0, i] = cx
        raw[0, 1, i] = cy
        raw[0, 2, i] = w
        raw[0, 3, i] = h
        raw[0, 4, i] = s
    return raw


def _ctx(raw: np.ndarray, card: FinwaveModelCard, orig_size: tuple[int, int]) -> PostprocessCtx:
    image = Image.new("RGB", orig_size, color=(0, 0, 0))
    return PostprocessCtx(outputs={"output0": raw}, card=card, state=None, image=image)


# ── decode + nms ──────────────────────────────────────────────────────────

def test_decode_filters_below_confidence() -> None:
    raw = _raw_v8([(320, 320, 100, 100), (320, 320, 100, 100)], [0.1, 0.9])
    boxes, scores, _ = _decode_v8(raw[0], conf_threshold=0.5)
    assert boxes.shape == (1, 4)
    assert float(scores[0]) == pytest.approx(0.9, abs=1e-6)


def test_decode_empty_when_all_below_threshold() -> None:
    raw = _raw_v8([(100, 100, 50, 50)], [0.1])
    boxes, scores, ids = _decode_v8(raw[0], conf_threshold=0.5)
    assert len(boxes) == 0 and len(scores) == 0 and len(ids) == 0


def test_nms_suppresses_overlapping_lower_score() -> None:
    # Two near-identical boxes; NMS keeps the higher-scoring one.
    boxes = np.array([[100, 100, 200, 200], [105, 105, 205, 205]], dtype=np.float32)
    scores = np.array([0.95, 0.80], dtype=np.float32)
    keep = _nms(boxes, scores, iou_threshold=0.5)
    assert len(keep) == 1
    assert int(keep[0]) == 0  # higher score wins


def test_nms_keeps_distant_boxes() -> None:
    boxes = np.array([[10, 10, 50, 50], [200, 200, 250, 250]], dtype=np.float32)
    scores = np.array([0.9, 0.85], dtype=np.float32)
    keep = _nms(boxes, scores, iou_threshold=0.5)
    assert sorted(keep.tolist()) == [0, 1]


# ── letterbox undo ────────────────────────────────────────────────────────

def test_undo_letterbox_landscape() -> None:
    # Original 1280x720 → letterbox to 640x640: scale=0.5, pad_y=140.
    boxes_model_space = np.array([[100, 200, 300, 400]], dtype=np.float32)
    out = _undo_letterbox(boxes_model_space, model_size=640, orig_size=(1280, 720))
    # (100 - 0) / 0.5 = 200; (200 - 140) / 0.5 = 120; (300 - 0)/0.5 = 600; (400 - 140)/0.5 = 520
    assert out[0].tolist() == [200.0, 120.0, 600.0, 520.0]


def test_undo_letterbox_square_no_pad() -> None:
    boxes = np.array([[50, 50, 150, 150]], dtype=np.float32)
    out = _undo_letterbox(boxes, model_size=640, orig_size=(640, 640))
    assert out[0].tolist() == [50.0, 50.0, 150.0, 150.0]


# ── end-to-end through the registered postprocess fns ─────────────────────

def test_yolo_pipeline_proportional_box() -> None:
    card = _make_card(image_size=640)
    # Original 800x600. Letterbox: scale=0.8, pad_y=(640-480)/2=80, pad_x=0.
    # Place a box at model-space (320, 320, 100, 100) - i.e., x1=270, y1=270, x2=370, y2=370.
    # Original-space after undo: x1=(270-0)/0.8=337.5, y1=(270-80)/0.8=237.5,
    #                            x2=(370-0)/0.8=462.5, y2=(370-80)/0.8=362.5
    # centre 400, 300 ; w 125, h 125 ; proportional: 0.5, 0.5, 0.15625, 0.20833...
    raw = _raw_v8([(320, 320, 100, 100)], [0.9])
    ctx = _ctx(raw, card, orig_size=(800, 600))

    prop = get("yolo_boxes_proportional")(ctx)
    assert len(prop) == 1
    assert prop[0]["X"] == 0.5
    assert prop[0]["Y"] == 0.5
    assert abs(prop[0]["W"] - 125 / 800) < 1e-6
    assert abs(prop[0]["H"] - 125 / 600) < 1e-6

    abs_boxes = get("yolo_boxes_absolute")(ctx)
    assert abs(abs_boxes[0]["X"] - 400.0) < 1e-3
    assert abs(abs_boxes[0]["Y"] - 300.0) < 1e-3

    confs = get("yolo_confidences")(ctx)
    assert confs == pytest.approx([0.9], abs=1e-6)


def test_yolo_pipeline_returns_crops() -> None:
    card = _make_card(image_size=640)
    raw = _raw_v8([(320, 320, 200, 200)], [0.95])
    ctx = _ctx(raw, card, orig_size=(640, 640))
    crops = get("yolo_crops_base64")(ctx)
    assert len(crops) == 1
    assert isinstance(crops[0], str) and len(crops[0]) > 0


def test_yolo_pipeline_empty_when_no_detections() -> None:
    card = _make_card(image_size=640)
    raw = _raw_v8([(320, 320, 50, 50)], [0.05])  # below conf_threshold=0.25
    ctx = _ctx(raw, card, orig_size=(640, 640))
    assert get("yolo_boxes_proportional")(ctx) == []
    assert get("yolo_confidences")(ctx) == []
    assert get("yolo_crops_base64")(ctx) == []


def test_postprocess_registry_lists_all_fns() -> None:
    from finwave_inference_server.postprocess import names

    expected = {
        "ncc_argmax_label",
        "ncc_softmax_max",
        "ncc_softmax_dict",
        "is_novel",
        "softmax_max_label",
        "softmax_max",
        "softmax_dict",
        "quality_score",
        "yolo_boxes_proportional",
        "yolo_boxes_absolute",
        "yolo_crops_base64",
        "yolo_confidences",
    }
    assert expected.issubset(set(names()))
