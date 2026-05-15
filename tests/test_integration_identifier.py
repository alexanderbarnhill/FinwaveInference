"""End-to-end integration test for an Identifier model.

Builds a tiny ONNX encoder (ReduceMean over spatial dims → MatMul) and a matching
classifier sidecar (3 classes, 512-d centroids), registers the model with the
running app, and fires an inference. Validates the full POST /models/register →
POST /inference path against the contract in `app/docs/architecture/contracts/backend-ml.md`.
"""
from __future__ import annotations

import base64
import hashlib
import io
import os
from pathlib import Path

os.environ.setdefault("FINWAVE_API_KEY", "test-key")

import numpy as np
import onnx
from fastapi.testclient import TestClient
from onnx import TensorProto, helper, numpy_helper
from PIL import Image

from finwave_inference_server.main import app

EMBEDDING_DIM = 64  # small, keeps the test fast


def _build_demo_encoder(path: Path) -> None:
    """Tiny ONNX encoder: input (1,3,224,224) → ReduceMean over HW → MatMul → embedding (1, EMBEDDING_DIM)."""
    rng = np.random.RandomState(42)
    W = (rng.randn(3, EMBEDDING_DIM) * 0.1).astype(np.float32)

    inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 224, 224])
    out = helper.make_tensor_value_info("embedding", TensorProto.FLOAT, [1, EMBEDDING_DIM])
    weight = numpy_helper.from_array(W, name="W")

    pool = helper.make_node(
        "ReduceMean", ["input"], ["pooled"], axes=[2, 3], keepdims=0
    )  # (1, 3)
    matmul = helper.make_node("MatMul", ["pooled", "W"], ["embedding"])

    graph = helper.make_graph([pool, matmul], "demo_encoder", [inp], [out], [weight])
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_operatorsetid("", 13)],
        producer_name="finwave-test",
    )
    onnx.checker.check_model(model)
    onnx.save(model, str(path))


def _build_demo_classifier(path: Path) -> None:
    """Synthetic NCC sidecar — 3 classes with fixed centroids."""
    rng = np.random.RandomState(123)
    classes = np.array([0, 1, 2], dtype=np.int64)
    centroids = rng.randn(3, EMBEDDING_DIM).astype(np.float32)
    np.savez(path, cls=classes, centroids=centroids)


def _build_demo_image(path: Path) -> None:
    Image.new("RGB", (224, 224), color=(128, 64, 200)).save(path, "JPEG")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _b64_image(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _card(encoder_path: Path, classifier_path: Path) -> dict:
    return {
        "spec_version": "1.0.0",
        "job_id": "integration-test-job",
        "model_name": "demo_identifier",
        "node_type": "Identifier",
        "input": {
            "contract": "Detection",
            "image_size": 224,
            "channels": 3,
            "normalization": "imagenet",
        },
        "output": {
            "contract": "IdentificationResult",
            "fields": [
                {"name": "Class", "type": "string", "source": "postprocess:ncc_argmax_label"},
                {"name": "Probability", "type": "float", "source": "postprocess:ncc_softmax_max"},
                {
                    "name": "Probabilities",
                    "type": "map<string,float>",
                    "source": "postprocess:ncc_softmax_dict",
                },
                {
                    "name": "Embedding",
                    "type": "float[]",
                    "dim": EMBEDDING_DIM,
                    "source": "onnx:embedding",
                },
            ],
            "produces_embedding": True,
            "embedding_dim": EMBEDDING_DIM,
            "probabilistic": True,
            "class_dict": {"0": "alpha", "1": "beta", "2": "gamma"},
        },
        "inference_config": {
            "distance_metric": "cosine",
            "temperature": 0.1,
        },
        "artifact": {
            "format": "onnx-bundle",
            "entrypoint": "encoder.onnx",
            "classifier_file": "classifier.npz",
            "files": [
                {
                    "name": "encoder.onnx",
                    "url": f"file://{encoder_path}",
                    "sha256": _sha256(encoder_path),
                },
                {
                    "name": "classifier.npz",
                    "url": f"file://{classifier_path}",
                    "sha256": _sha256(classifier_path),
                },
            ],
        },
    }


def test_register_and_infer_identifier_end_to_end(tmp_path: Path) -> None:
    encoder = tmp_path / "encoder.onnx"
    classifier = tmp_path / "classifier.npz"
    image = tmp_path / "test.jpg"

    _build_demo_encoder(encoder)
    _build_demo_classifier(classifier)
    _build_demo_image(image)

    # The registry caches downloads under model_store_path; redirect to tmp.
    os.environ["FINWAVE_MODEL_STORE_PATH"] = str(tmp_path / "store")
    from finwave_inference_server.config import get_settings
    get_settings.cache_clear()  # type: ignore[attr-defined]

    card = _card(encoder, classifier)

    with TestClient(app) as client:
        # 1. Register the model
        resp = client.post(
            "/models/register",
            headers={"X-API-KEY": "test-key"},
            json={"model_card": card},
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["model_name"] == "demo_identifier"

        # 2. List shows it
        resp = client.get("/models", headers={"X-API-KEY": "test-key"})
        models = resp.json()
        assert len(models) == 1
        assert models[0]["model_name"] == "demo_identifier"
        assert models[0]["node_type"] == "Identifier"

        # 3. Health reflects the load count
        resp = client.get("/health")
        assert resp.json()["models_loaded"] == 1

        # 4. Inference end-to-end
        resp = client.post(
            "/inference",
            json={"model_name": "demo_identifier", "image": _b64_image(image)},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()

        # Every declared output field is present
        assert set(body.keys()) == {"Class", "Probability", "Probabilities", "Embedding"}

        # Class is one of the three declared labels
        assert body["Class"] in {"alpha", "beta", "gamma"}

        # Probability is the max of Probabilities
        assert isinstance(body["Probability"], float)
        assert 0.0 <= body["Probability"] <= 1.0
        assert abs(body["Probability"] - max(body["Probabilities"].values())) < 1e-6

        # Probabilities sum to ~1 across the three classes
        assert set(body["Probabilities"].keys()) == {"alpha", "beta", "gamma"}
        assert abs(sum(body["Probabilities"].values()) - 1.0) < 1e-5

        # Embedding has the declared dim
        assert isinstance(body["Embedding"], list)
        assert len(body["Embedding"]) == EMBEDDING_DIM
        assert all(isinstance(v, float) for v in body["Embedding"])


def test_register_rejects_mar_format(tmp_path: Path) -> None:
    """Registration with format=mar should be rejected — adapter not implemented yet."""
    encoder = tmp_path / "x.mar"
    encoder.write_bytes(b"not a real MAR")

    card = {
        "spec_version": "1.0.0",
        "job_id": "j",
        "model_name": "legacy",
        "node_type": "Identifier",
        "input": {"contract": "Detection", "image_size": 224},
        "output": {
            "contract": "IdentificationResult",
            "fields": [{"name": "Class", "type": "string", "source": "postprocess:ncc_argmax_label"}],
        },
        "artifact": {
            "format": "mar",
            "entrypoint": "x.mar",
            "files": [{"name": "x.mar", "url": f"file://{encoder}", "sha256": _sha256(encoder)}],
        },
    }

    with TestClient(app) as client:
        resp = client.post(
            "/models/register",
            headers={"X-API-KEY": "test-key"},
            json={"model_card": card},
        )
        assert resp.status_code == 400
        assert "MAR adapter not implemented" in resp.text


def test_register_rejects_unknown_postprocess(tmp_path: Path) -> None:
    """A card referencing a nonexistent postprocess fn should 400."""
    encoder = tmp_path / "encoder.onnx"
    _build_demo_encoder(encoder)

    card = {
        "spec_version": "1.0.0",
        "job_id": "j",
        "model_name": "bad-postprocess",
        "node_type": "Identifier",
        "input": {"contract": "Detection", "image_size": 224},
        "output": {
            "contract": "IdentificationResult",
            "fields": [{"name": "X", "type": "string", "source": "postprocess:does_not_exist"}],
        },
        "artifact": {
            "format": "onnx-bundle",
            "entrypoint": "encoder.onnx",
            "files": [{"name": "encoder.onnx", "url": f"file://{encoder}", "sha256": _sha256(encoder)}],
        },
    }

    with TestClient(app) as client:
        resp = client.post(
            "/models/register",
            headers={"X-API-KEY": "test-key"},
            json={"model_card": card},
        )
        assert resp.status_code == 400
        assert "does_not_exist" in resp.text
