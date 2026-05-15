"""Smoke test: app imports, health endpoint responds, unauthenticated /models returns 401."""
import os

os.environ.setdefault("FINWAVE_API_KEY", "test-key")

from fastapi.testclient import TestClient

from finwave_inference_server.main import app


def test_health_returns_ok() -> None:
    with TestClient(app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["models_loaded"] == 0


def test_models_list_requires_api_key() -> None:
    with TestClient(app) as client:
        resp = client.get("/models")
        assert resp.status_code == 401


def test_models_list_accepts_api_key() -> None:
    with TestClient(app) as client:
        resp = client.get("/models", headers={"X-API-KEY": "test-key"})
        assert resp.status_code == 200
        assert resp.json() == []
