"""#6 — request size cap (body + per-image) and the readiness probe."""
from __future__ import annotations

import os

os.environ.setdefault("FINWAVE_API_KEY", "test-key")

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from finwave_inference_server.config import get_settings
from finwave_inference_server.main import _ensure_image_within_limit, app


def test_ready_is_503_when_no_models_loaded() -> None:
    # /health is always 200 while the process is up; /ready gates on having a model loaded.
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/ready").status_code == 503


def test_ensure_image_within_limit(monkeypatch) -> None:
    monkeypatch.setenv("FINWAVE_MAX_IMAGE_MB", "1")
    get_settings.cache_clear()
    try:
        _ensure_image_within_limit("A" * 100)  # tiny → allowed
        with pytest.raises(HTTPException) as ei:
            _ensure_image_within_limit("A" * (2 * 1024 * 1024))  # ~1.5 MB decoded > 1 MB cap
        assert ei.value.status_code == 413
    finally:
        get_settings.cache_clear()


def test_inference_rejects_oversized_request(monkeypatch) -> None:
    monkeypatch.setenv("FINWAVE_MAX_IMAGE_MB", "1")
    get_settings.cache_clear()
    try:
        with TestClient(app) as client:
            resp = client.post(
                "/inference",
                json={"model_name": "nope", "image": "A" * (3 * 1024 * 1024)},
            )
            assert resp.status_code == 413
    finally:
        get_settings.cache_clear()
