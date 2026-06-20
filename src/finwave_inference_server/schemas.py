"""Runtime-relevant subset of FinwaveModelCard + request/response shapes.

This is a local copy. Once the shared `finwave-contracts` package exists,
delete this file and import from there. The fields here MUST match the
spec in `app/docs/architecture/contracts/backend-ml.md`.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict


class _Base(BaseModel):
    model_config = ConfigDict(protected_namespaces=())


class ArtifactFile(_Base):
    name: str
    url: str
    sha256: str


class ArtifactSpec(_Base):
    format: Literal["onnx-bundle", "mar"]
    entrypoint: str
    files: list[ArtifactFile]
    classifier_file: str | None = None


class FieldSpec(_Base):
    name: str
    type: str
    source: str
    dim: int | None = None


class InputSpec(_Base):
    contract: str
    image_size: int
    channels: int = 3
    normalization: str | dict[str, list[float]] = "imagenet"


class OutputSpec(_Base):
    contract: str
    fields: list[FieldSpec]
    produces_embedding: bool = False
    embedding_dim: int | None = None
    probabilistic: bool = True
    class_dict: dict[str, str] | None = None


class InferenceConfig(_Base):
    # Identifier / Classifier (NCC) params
    distance_metric: Literal["cosine", "euclidean"] = "cosine"
    temperature: float = 1.0
    novelty_threshold: float | None = None
    novelty_threshold_99: float | None = None
    sub_centers: int = 0
    # Detector (YOLO) params
    conf_threshold: float = 0.15
    iou_threshold: float = 0.5
    crop_format: Literal["JPEG", "PNG"] = "JPEG"


NodeType = Literal[
    "Detector",
    "Validator",
    "SideClassifier",
    "Identifier",
    "QualityScorer",
    "Classifier",
]


class FinwaveModelCard(_Base):
    spec_version: str
    job_id: str
    model_name: str
    population_id: str | None = None
    node_type: NodeType
    replaces_model_id: str | None = None

    input: InputSpec
    output: OutputSpec
    inference_config: InferenceConfig = InferenceConfig()
    artifact: ArtifactSpec
    # Non-runtime metadata, preserved end-to-end so a deployed model stays
    # self-describing: where it came from (checkpoint, dataset, backbone, git
    # SHA, calibration source) and how it scored (held-out rank-1/5). Optional
    # for backward compatibility with older cards.
    provenance: dict[str, Any] | None = None
    evaluation: dict[str, Any] | None = None


class InferenceRequest(_Base):
    model_name: str
    image: str  # base64-encoded PNG/JPEG


class RegisterModelRequest(_Base):
    model_card: FinwaveModelCard
