"""NCC postprocess functions for Identifier / Classifier models.

Mirrors the math in toolkit/metric/deploy/handler.py — distance to centroids,
temperature-scaled softmax, sub-center support, novelty thresholding.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ._registry import register


def _pairwise(a: np.ndarray, b: np.ndarray, metric: str) -> np.ndarray:
    """Distance between every row of `a` and every row of `b`."""
    if metric == "cosine":
        an = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-12)
        bn = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-12)
        return 1.0 - an @ bn.T
    if metric == "euclidean":
        return np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)
    raise ValueError(f"unknown distance metric: {metric!r}")


def _distances(
    embedding: np.ndarray, state: dict[str, Any], metric: str
) -> tuple[np.ndarray, np.ndarray]:
    """Return (per-class min distance, class labels)."""
    classes = state["cls"]
    emb = embedding.reshape(1, -1)
    if "sub_center_weights" in state:
        scw = state["sub_center_weights"]  # (num_classes, sub_centers, dim)
        num_classes, k, dim = scw.shape
        all_d = _pairwise(emb, scw.reshape(-1, dim), metric)[0]
        per_class = all_d.reshape(num_classes, k).min(axis=1)
    else:
        per_class = _pairwise(emb, state["centroids"], metric)[0]
    return per_class, classes


def _softmax_probs(distances: np.ndarray, temperature: float) -> np.ndarray:
    logits = -distances / max(temperature, 1e-12)
    logits = logits - logits.max()
    exp = np.exp(logits)
    return exp / exp.sum()


def _label(card: Any, key: Any) -> str:
    cd = card.output.class_dict or {}
    k = str(key)
    return cd.get(k, k)


@register("ncc_argmax_label")
def ncc_argmax_label(onnx_outputs, card, state):
    distances, classes = _distances(
        onnx_outputs["embedding"], state, card.inference_config.distance_metric
    )
    return _label(card, classes[int(np.argmin(distances))])


@register("ncc_softmax_max")
def ncc_softmax_max(onnx_outputs, card, state):
    distances, _ = _distances(
        onnx_outputs["embedding"], state, card.inference_config.distance_metric
    )
    probs = _softmax_probs(distances, card.inference_config.temperature)
    return float(probs.max())


@register("ncc_softmax_dict")
def ncc_softmax_dict(onnx_outputs, card, state):
    distances, classes = _distances(
        onnx_outputs["embedding"], state, card.inference_config.distance_metric
    )
    probs = _softmax_probs(distances, card.inference_config.temperature)
    return {_label(card, c): float(p) for c, p in zip(classes, probs)}


@register("is_novel")
def is_novel(onnx_outputs, card, state):
    distances, _ = _distances(
        onnx_outputs["embedding"], state, card.inference_config.distance_metric
    )
    threshold = card.inference_config.novelty_threshold
    if threshold is None:
        return False
    return bool(float(distances.min()) > threshold)
