"""Postprocess library. Functions register themselves at import time."""
from ._registry import get, names, register

from . import ncc, passthrough, yolo  # noqa: F401  (registration side-effects)

__all__ = ["get", "names", "register"]
