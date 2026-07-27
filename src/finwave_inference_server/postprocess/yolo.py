"""YOLOv8-style postprocess for Detector models.

The toolkit's `pipeline_detect/export.py` exports YOLO models with the standard
Ultralytics ONNX layout: single output of shape `(batch, 4+num_classes, num_anchors)`
in model-input-space coords (after letterbox padding).

This module decodes that raw tensor to bounding boxes in the original image's
pixel space, runs NMS, and exposes four fields callers can wire from
ModelCard.output.fields[].source:

  - `yolo_boxes_proportional`  → [{X, Y, W, H}] in [0,1] (top-left origin)
  - `yolo_boxes_absolute`      → [{X, Y, W, H}] in pixels (top-left origin)
  - `yolo_crops_base64`        → [base64-encoded JPEG/PNG crops of each box]
  - `yolo_confidences`         → [float] one score per kept box

Thresholds come from `inference_config.conf_threshold` (default 0.15) and
`inference_config.iou_threshold` (default 0.5); crop format from `crop_format`.
"""
from __future__ import annotations

import base64
import io

import numpy as np

from ._registry import PostprocessCtx, register


def _decode_v8(
    raw: np.ndarray, conf_threshold: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode YOLOv8 raw output to filtered (boxes_xyxy, scores, class_ids).

    `raw` shape is `(4+num_classes, num_anchors)` — the (batch=1) dim already removed.
    Returned boxes are in model-input-space (before letterbox-undo).
    """
    raw = raw.T  # (anchors, 4+nc)
    if raw.shape[1] < 5:
        raise ValueError(f"expected raw shape (..., 4+num_classes); got {raw.shape}")

    bboxes = raw[:, :4]
    cls_scores = raw[:, 4:]
    scores = cls_scores.max(axis=1)
    class_ids = cls_scores.argmax(axis=1).astype(int)

    keep = scores >= conf_threshold
    bboxes = bboxes[keep]
    scores = scores[keep]
    class_ids = class_ids[keep]
    if len(bboxes) == 0:
        return np.empty((0, 4), dtype=np.float32), scores, class_ids

    cx, cy, w, h = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
    boxes_xyxy = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
    return boxes_xyxy, scores, class_ids


def _decode_e2e(
    raw: np.ndarray, conf_threshold: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode an end-to-end / NMS-free detector (YOLOv10/YOLO26) output.

    Layout is `(num_detections, 6)` = `[x1, y1, x2, y2, score, class]`, already
    xyxy in model-input space and already NMS-filtered, so we only threshold.
    """
    if raw.shape[1] != 6:
        raise ValueError(f"expected e2e shape (num_det, 6); got {raw.shape}")
    keep = raw[:, 4] >= conf_threshold
    r = raw[keep]
    return (r[:, :4].astype(np.float32), r[:, 4].astype(np.float32), r[:, 5].astype(int))


def _nms(boxes_xyxy: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
    """Non-maximum suppression. Returns indices to keep, sorted by score desc."""
    if len(boxes_xyxy) == 0:
        return np.empty(0, dtype=int)
    x1, y1, x2, y2 = boxes_xyxy[:, 0], boxes_xyxy[:, 1], boxes_xyxy[:, 2], boxes_xyxy[:, 3]
    areas = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while len(order) > 0:
        i = int(order[0])
        keep.append(i)
        if len(order) == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.clip(xx2 - xx1, 0, None)
        h = np.clip(yy2 - yy1, 0, None)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-12)
        order = order[1:][iou < iou_threshold]
    return np.array(keep, dtype=int)


def _undo_letterbox(
    boxes_xyxy: np.ndarray, model_size: int, orig_size: tuple[int, int]
) -> np.ndarray:
    """Map boxes from letterboxed model-input-space back to original image pixels.

    Mirrors the letterbox in registry._resize_letterbox: scale so max(w,h) == model_size,
    then centre-pad to (model_size, model_size).
    """
    orig_w, orig_h = orig_size
    scale = model_size / max(orig_w, orig_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)
    pad_x = (model_size - new_w) // 2
    pad_y = (model_size - new_h) // 2

    out = boxes_xyxy.astype(np.float32, copy=True)
    out[:, [0, 2]] -= pad_x
    out[:, [1, 3]] -= pad_y
    out /= scale
    out[:, [0, 2]] = np.clip(out[:, [0, 2]], 0, orig_w)
    out[:, [1, 3]] = np.clip(out[:, [1, 3]], 0, orig_h)
    return out


def _all_boxes(ctx: PostprocessCtx) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the full pipeline once per request; cache on the ctx for the four field fns."""
    cache = getattr(ctx, "_yolo_cache", None)
    if cache is not None:
        return cache  # type: ignore[return-value]

    cfg = ctx.card.inference_config
    output_names = list(ctx.outputs.keys())
    if not output_names:
        raise ValueError("YOLO postprocess: model produced no outputs")
    raw = ctx.outputs[output_names[0]]
    if raw.ndim == 3:
        raw = raw[0]

    # End-to-end detectors (YOLOv10/26) emit (num_det, 6); classic YOLOv8 emits
    # (4+nc, anchors) where the second axis is thousands of anchors. The e2e
    # output is already NMS-filtered, so skip NMS for it.
    if raw.ndim == 2 and raw.shape[1] == 6:
        boxes, scores, class_ids = _decode_e2e(raw, cfg.conf_threshold)
    else:
        boxes, scores, class_ids = _decode_v8(raw, cfg.conf_threshold)
        keep = _nms(boxes, scores, cfg.iou_threshold)
        boxes = boxes[keep]
        scores = scores[keep]
        class_ids = class_ids[keep]

    boxes_orig = _undo_letterbox(boxes, ctx.card.input.image_size, ctx.image.size)
    result = (boxes_orig, scores, class_ids)
    ctx._yolo_cache = result  # type: ignore[attr-defined]
    return result


@register("yolo_boxes_proportional")
def yolo_boxes_proportional(ctx: PostprocessCtx):
    boxes_xyxy, _, _ = _all_boxes(ctx)
    orig_w, orig_h = ctx.image.size
    out = []
    for x1, y1, x2, y2 in boxes_xyxy:
        w = x2 - x1
        h = y2 - y1
        # X,Y are the box's TOP-LEFT corner (finwave annotations use top-left origin,
        # matching human-drawn Konva boxes). Reporting the centre here shifted every
        # machine box down-right by (W/2, H/2).
        out.append(
            {
                "X": float(x1) / orig_w,
                "Y": float(y1) / orig_h,
                "W": float(w) / orig_w,
                "H": float(h) / orig_h,
            }
        )
    return out


@register("yolo_boxes_absolute")
def yolo_boxes_absolute(ctx: PostprocessCtx):
    boxes_xyxy, scores, _ = _all_boxes(ctx)
    out = []
    for (x1, y1, x2, y2), score in zip(boxes_xyxy, scores):
        w = x2 - x1
        h = y2 - y1
        # X,Y = top-left corner (see yolo_boxes_proportional).
        out.append(
            {
                "X": float(x1),
                "Y": float(y1),
                "W": float(w),
                "H": float(h),
                # Per-box confidence — the worker's DetectorHandler reads
                # AbsoluteBoxes[i]["Confidence"] and defaults to 1.0 when absent,
                # so without this every ML annotation would be Probability=1.0.
                "Confidence": float(score),
            }
        )
    return out


@register("yolo_crops_base64")
def yolo_crops_base64(ctx: PostprocessCtx):
    boxes_xyxy, _, _ = _all_boxes(ctx)
    fmt = ctx.card.inference_config.crop_format
    crops = []
    for x1, y1, x2, y2 in boxes_xyxy:
        crop = ctx.image.crop((float(x1), float(y1), float(x2), float(y2)))
        if crop.width < 1 or crop.height < 1:
            continue
        buf = io.BytesIO()
        crop.save(buf, format=fmt)
        crops.append(base64.b64encode(buf.getvalue()).decode("utf-8"))
    return crops


@register("yolo_confidences")
def yolo_confidences(ctx: PostprocessCtx):
    _, scores, _ = _all_boxes(ctx)
    return [float(s) for s in scores]
