"""Decorator-based registry for postprocess functions referenced from ModelCard.output.fields[].source."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage


@dataclass
class PostprocessCtx:
    """Everything a postprocess fn may need.

    Fields:
        outputs: dict of ONNX output name -> ndarray returned by the session.
        card: the FinwaveModelCard for the model being served.
        state: optional classifier sidecar (NCC centroids, normalization stats, ...).
                Loaded from `card.artifact.classifier_file`.
        image: the original (pre-letterbox) PIL image. Detectors need this for crop
               extraction and letterbox-undo; classifiers ignore it.
    """

    outputs: dict[str, np.ndarray]
    card: Any
    state: dict[str, Any] | None
    image: "PILImage"


PostprocessFn = Callable[[PostprocessCtx], Any]

_REGISTRY: dict[str, PostprocessFn] = {}


def register(name: str) -> Callable[[PostprocessFn], PostprocessFn]:
    def wrap(fn: PostprocessFn) -> PostprocessFn:
        if name in _REGISTRY:
            raise ValueError(f"postprocess {name!r} already registered")
        _REGISTRY[name] = fn
        return fn

    return wrap


def get(name: str) -> PostprocessFn:
    if name not in _REGISTRY:
        raise KeyError(f"postprocess {name!r} not registered; available: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def names() -> list[str]:
    return sorted(_REGISTRY)
