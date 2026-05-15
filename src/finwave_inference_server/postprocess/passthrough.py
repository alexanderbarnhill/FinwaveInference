"""Generic postprocess functions for non-NCC paradigms (Validator, SideClassifier, QualityScorer)."""
from __future__ import annotations

import numpy as np

from ._registry import register


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


@register("softmax_max_label")
def softmax_max_label(onnx_outputs, card, state):
    logits = onnx_outputs["logits"].reshape(-1)
    idx = int(np.argmax(_softmax(logits)))
    cd = card.output.class_dict or {}
    return cd.get(str(idx), str(idx))


@register("softmax_max")
def softmax_max(onnx_outputs, card, state):
    return float(_softmax(onnx_outputs["logits"].reshape(-1)).max())


@register("softmax_dict")
def softmax_dict(onnx_outputs, card, state):
    probs = _softmax(onnx_outputs["logits"].reshape(-1))
    cd = card.output.class_dict or {}
    return {cd.get(str(i), str(i)): float(p) for i, p in enumerate(probs)}


@register("quality_score")
def quality_score(onnx_outputs, card, state):
    raw = float(onnx_outputs["score"].reshape(-1)[0])
    if state and "mean" in state and "std" in state:
        return (raw - float(state["mean"])) / max(float(state["std"]), 1e-12)
    return raw
