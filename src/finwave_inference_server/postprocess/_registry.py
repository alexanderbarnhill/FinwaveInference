"""Decorator-based registry for postprocess functions referenced from ModelCard.output.fields[].source."""
from typing import Any, Callable

import numpy as np

PostprocessFn = Callable[[dict[str, np.ndarray], Any, Any], Any]
"""Signature: (onnx_outputs, model_card, classifier_state) -> any."""

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
        raise KeyError(
            f"postprocess {name!r} not registered; available: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name]


def names() -> list[str]:
    return sorted(_REGISTRY)
