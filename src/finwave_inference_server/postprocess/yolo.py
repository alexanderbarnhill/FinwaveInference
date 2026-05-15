"""YOLO postprocess functions for Detector models.

Pending: port from toolkit/fin_detect/deploy/handler.py and
toolkit/feature_detect/deploy/handler.py. Stubs are registered so
ModelCards referencing them validate; calling them at inference time
raises NotImplementedError until the port lands.
"""
from ._registry import register


@register("yolo_boxes_proportional")
def yolo_boxes_proportional(onnx_outputs, card, state):
    raise NotImplementedError(
        "yolo_boxes_proportional pending — port from "
        "toolkit/fin_detect/deploy/handler.py"
    )


@register("yolo_boxes_absolute")
def yolo_boxes_absolute(onnx_outputs, card, state):
    raise NotImplementedError("yolo_boxes_absolute pending — see yolo_boxes_proportional")


@register("yolo_crops_base64")
def yolo_crops_base64(onnx_outputs, card, state):
    raise NotImplementedError("yolo_crops_base64 pending — see yolo_boxes_proportional")
