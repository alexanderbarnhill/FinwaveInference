"""Fetch a model bundle declared by a FinwaveModelCard and load it into a runtime session."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import onnxruntime as ort

from .schemas import FinwaveModelCard


async def fetch_and_verify(card: FinwaveModelCard, dest_dir: Path) -> dict[str, Path]:
    """Download every file in `card.artifact.files` into `dest_dir`.

    Returns `{file_name: local_path}`. Raises if any sha256 fails.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    async with httpx.AsyncClient(timeout=300.0, follow_redirects=True) as client:
        for entry in card.artifact.files:
            path = dest_dir / entry.name
            await _download_to(client, entry.url, path)
            digest = _sha256(path)
            if digest != entry.sha256:
                path.unlink(missing_ok=True)
                raise ValueError(
                    f"sha256 mismatch for {entry.name}: "
                    f"expected {entry.sha256}, got {digest}"
                )
            paths[entry.name] = path
    return paths


async def _download_to(client: httpx.AsyncClient, url: str, dest: Path) -> None:
    async with client.stream("GET", url) as resp:
        resp.raise_for_status()
        with dest.open("wb") as fp:
            async for chunk in resp.aiter_bytes(1024 * 1024):
                fp.write(chunk)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_onnx_session(entrypoint: Path) -> ort.InferenceSession:
    providers = ort.get_available_providers()
    return ort.InferenceSession(str(entrypoint), providers=providers)


def load_classifier_state(
    card: FinwaveModelCard, files: dict[str, Path]
) -> dict[str, Any] | None:
    """Load the sidecar referenced by `card.artifact.classifier_file`.

    Returns None when no classifier file is declared.
    """
    if card.artifact.classifier_file is None:
        return None
    path = files[card.artifact.classifier_file]
    if path.suffix == ".npz":
        data = np.load(path, allow_pickle=False)
        return {key: data[key] for key in data.files}
    if path.suffix == ".json":
        return json.loads(path.read_text())
    raise ValueError(f"unsupported classifier_file extension: {path.suffix}")
