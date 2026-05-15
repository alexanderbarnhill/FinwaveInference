"""In-memory registry of loaded models.

One `LoadedModel` per registered ModelCard. Registration: fetch + verify
artifact files → load onnxruntime session → cache classifier state. Inference:
preprocess → session.run → assemble response from declared `output.fields`.
"""
from __future__ import annotations

import asyncio
import base64
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from PIL import Image

from .config import Settings
from .loader import fetch_and_verify, load_classifier_state, load_onnx_session
from .postprocess import get as get_postprocess
from .postprocess._registry import PostprocessCtx
from .schemas import FinwaveModelCard


_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


@dataclass
class LoadedModel:
    card: FinwaveModelCard
    session: ort.InferenceSession
    classifier_state: dict[str, Any] | None
    files: dict[str, Path]


class ModelRegistry:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._models: dict[str, LoadedModel] = {}

    def __len__(self) -> int:
        return len(self._models)

    def list(self) -> list[dict]:
        return [
            {
                "model_name": name,
                "node_type": loaded.card.node_type,
                "spec_version": loaded.card.spec_version,
                "population_id": loaded.card.population_id,
            }
            for name, loaded in self._models.items()
        ]

    def get(self, name: str) -> LoadedModel:
        if name not in self._models:
            raise KeyError(f"model {name!r} not loaded")
        return self._models[name]

    async def register(self, card: FinwaveModelCard) -> None:
        major = int(card.spec_version.split(".", 1)[0])
        if major != self._settings.supported_spec_major:
            raise ValueError(
                f"spec_version major {major} not supported "
                f"(server supports {self._settings.supported_spec_major})"
            )
        if card.artifact.format == "mar":
            raise ValueError("MAR adapter not implemented yet; see backend-ml.md migration plan")

        for field in card.output.fields:
            prefix, _, ref = field.source.partition(":")
            if prefix == "postprocess":
                get_postprocess(ref)  # raises KeyError if unknown

        dest = Path(self._settings.model_store_path) / card.model_name
        files = await fetch_and_verify(card, dest)
        session = load_onnx_session(files[card.artifact.entrypoint])
        classifier_state = load_classifier_state(card, files)
        self._models[card.model_name] = LoadedModel(card, session, classifier_state, files)

    async def run(self, model_name: str, image_b64: str) -> dict[str, Any]:
        loaded = self.get(model_name)
        original, tensor = _preprocess(image_b64, loaded.card)
        outputs = await asyncio.to_thread(_run_session, loaded.session, tensor)
        return _assemble(outputs, original, loaded)


def _preprocess(image_b64: str, card: FinwaveModelCard) -> tuple[Image.Image, np.ndarray]:
    """Decode + letterbox-pad + normalize. Returns (original PIL, NCHW tensor)."""
    original = Image.open(io.BytesIO(base64.b64decode(image_b64))).convert("RGB")
    size = card.input.image_size
    letterboxed = _resize_letterbox(original, size)
    arr = np.asarray(letterboxed, dtype=np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)  # HWC → CHW
    arr = _normalize(arr, card.input.normalization)
    return original, arr[None, ...]


def _resize_letterbox(img: Image.Image, size: int) -> Image.Image:
    ratio = size / max(img.size)
    new_size = tuple(int(x * ratio) for x in img.size)
    img = img.resize(new_size, Image.LANCZOS)
    canvas = Image.new("RGB", (size, size))
    canvas.paste(img, ((size - new_size[0]) // 2, (size - new_size[1]) // 2))
    return canvas


def _normalize(arr: np.ndarray, norm: Any) -> np.ndarray:
    if norm == "imagenet":
        return (arr - _IMAGENET_MEAN) / _IMAGENET_STD
    if norm == "none":
        return arr
    if isinstance(norm, dict):
        mean = np.array(norm["mean"], dtype=np.float32).reshape(3, 1, 1)
        std = np.array(norm["std"], dtype=np.float32).reshape(3, 1, 1)
        return (arr - mean) / std
    raise ValueError(f"unsupported normalization: {norm!r}")


def _run_session(
    session: ort.InferenceSession, tensor: np.ndarray
) -> dict[str, np.ndarray]:
    input_name = session.get_inputs()[0].name
    output_names = [o.name for o in session.get_outputs()]
    raw = session.run(output_names, {input_name: tensor})
    return dict(zip(output_names, raw))


def _assemble(
    outputs: dict[str, np.ndarray], image: Image.Image, loaded: LoadedModel
) -> dict[str, Any]:
    ctx = PostprocessCtx(
        outputs=outputs,
        card=loaded.card,
        state=loaded.classifier_state,
        image=image,
    )
    response: dict[str, Any] = {}
    for field in loaded.card.output.fields:
        prefix, _, ref = field.source.partition(":")
        if prefix == "onnx":
            value = outputs[ref]
            response[field.name] = (
                value.flatten().tolist()
                if field.type.endswith("[]") or field.type.startswith("float[")
                else value.item()
            )
        elif prefix == "postprocess":
            fn = get_postprocess(ref)
            response[field.name] = fn(ctx)
        else:
            raise ValueError(f"unsupported source prefix: {prefix!r}")
    return response
