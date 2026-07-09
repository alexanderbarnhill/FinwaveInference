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
    if url.startswith("file://"):
        src = Path(url[len("file://") :])
        if not src.is_absolute():
            raise ValueError(f"file:// URL must be absolute: {url}")
        with src.open("rb") as inp, dest.open("wb") as out:
            while True:
                chunk = inp.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
        return
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


_dlls_preloaded = False


def _ensure_cuda_libs() -> None:
    """Load the CUDA/cuDNN shared libs shipped as nvidia-*-cu12 pip wheels.

    onnxruntime-gpu ships the CUDA execution provider but not the CUDA runtime;
    when those libs come from pip wheels (not a system CUDA install) they are not
    on the loader path, so the CUDA EP fails to initialise. `preload_dlls()` loads
    them from the installed wheels. No-op on CPU-only installs / older ORT.
    """
    global _dlls_preloaded
    if not _dlls_preloaded and hasattr(ort, "preload_dlls"):
        try:
            ort.preload_dlls()
        except Exception:  # best-effort — fall through to whatever EPs load
            pass
    _dlls_preloaded = True


def _preferred_providers() -> list[str]:
    """CUDA first, then CPU. Drop TensorRT: listing it without the TRT runtime
    makes ORT raise and fall back to CPU-only, silently skipping CUDA too."""
    available = set(ort.get_available_providers())
    return [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in available]


def load_onnx_session(entrypoint: Path) -> ort.InferenceSession:
    _ensure_cuda_libs()
    return ort.InferenceSession(str(entrypoint), providers=_preferred_providers())


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
