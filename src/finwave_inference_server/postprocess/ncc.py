"""NCC postprocess functions for Identifier / Classifier models.

Mirrors the math in toolkit/metric/deploy/handler.py — distance to centroids,
temperature-scaled softmax, sub-center support, novelty thresholding.
"""
from __future__ import annotations

import numpy as np

from ._registry import PostprocessCtx, register


def _pairwise(a: np.ndarray, b: np.ndarray, metric: str) -> np.ndarray:
    """Distance between every row of `a` and every row of `b`."""
    if metric == "cosine":
        an = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-12)
        bn = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-12)
        return 1.0 - an @ bn.T
    if metric == "euclidean":
        return np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)
    raise ValueError(f"unknown distance metric: {metric!r}")


def _distances(ctx: PostprocessCtx) -> tuple[np.ndarray, np.ndarray]:
    """Return (per-class min distance, class labels)."""
    state = ctx.state
    if state is None:
        raise ValueError("NCC postprocess requires a classifier sidecar (no state loaded)")
    classes = state["cls"]
    emb = ctx.outputs["embedding"].reshape(1, -1)
    metric = ctx.card.inference_config.distance_metric
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


def _label(card, key) -> str:
    cd = card.output.class_dict or {}
    k = str(key)
    return cd.get(k, k)


@register("ncc_argmax_label")
def ncc_argmax_label(ctx: PostprocessCtx):
    distances, classes = _distances(ctx)
    return _label(ctx.card, classes[int(np.argmin(distances))])


@register("ncc_softmax_max")
def ncc_softmax_max(ctx: PostprocessCtx):
    distances, _ = _distances(ctx)
    probs = _softmax_probs(distances, ctx.card.inference_config.temperature)
    return float(probs.max())


@register("ncc_softmax_dict")
def ncc_softmax_dict(ctx: PostprocessCtx):
    distances, classes = _distances(ctx)
    probs = _softmax_probs(distances, ctx.card.inference_config.temperature)
    return {_label(ctx.card, c): float(p) for c, p in zip(classes, probs)}


@register("is_novel")
def is_novel(ctx: PostprocessCtx):
    distances, _ = _distances(ctx)
    threshold = ctx.card.inference_config.novelty_threshold
    if threshold is None:
        return False
    return bool(float(distances.min()) > threshold)
