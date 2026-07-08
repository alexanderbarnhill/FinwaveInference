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


def test_legacy_inference_alias_routes_to_registry() -> None:
    """The finwave-inference-worker posts {Api, Image} to /api/inference. The
    alias must accept that body (not 422) and route by model name — 404 here
    because no model is loaded, which proves it reached registry.run."""
    with TestClient(app) as client:
        resp = client.post("/api/inference", json={"Api": "no-such-model", "Image": "Zm9v"})
        assert resp.status_code == 404


def test_legacy_inference_alias_rejects_wrong_body() -> None:
    with TestClient(app) as client:
        resp = client.post("/api/inference", json={"model_name": "x", "image": "y"})
        assert resp.status_code == 422
