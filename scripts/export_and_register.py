#!/usr/bin/env python3
"""Export a trained model to ONNX (detectors) and register it with a running
FinWave inference server — for every node type in the default pipeline.

Node types + what each needs (the card's `model_name` MUST equal the pipeline
node's `modelApi`):

  Detector        — an Ultralytics YOLO `.pt` (auto-exported) or an `.onnx`.
                    Emits ExtractedImages / ProportionBoxes / AbsoluteBoxes / Confidences.
  Validator       \
  SideClassifier   } — an `.onnx` whose output tensor is named `logits`, plus a
  Classifier      /    class list (--class-dict). Emits Class / Probability / Probabilities.
  Identifier      — a metric/NCC model: an `.onnx` whose output is named `embedding`,
                    an `.npz` sidecar (--classifier-file) with `cls` + `centroids`
                    (+ optional `sub_center_weights`), and a class list. Emits
                    Class / Probability / Probabilities / Embedding. See the toolkit
                    exporter `metric/deploy/export_onnx.py` which produces both files.

Examples
--------
Dorsal detector (YOLO .pt), registered under the pipeline's modelApi:
    uv run --with ultralytics --with httpx python scripts/export_and_register.py \
      --weights /media/alex/Storage/finwave_models/dorsal_fin/best.pt \
      --model-name FIN_DETECT --node-type Detector --img-size 640 --api-key "$FINWAVE_API_KEY"

Side classifier (already exported to ONNX with a `logits` output):
    ... --onnx fin_side.onnx --model-name FIN_SIDE --node-type SideClassifier \
        --class-dict '{"0":"left","1":"right"}' --img-size 224

Identifier (metric/NCC, exported via the toolkit exporter):
    ... --onnx embedding.onnx --model-name BKWPID-A_eB0 --node-type Identifier \
        --classifier-file ncc.npz --class-dict class_dict.json --img-size 512 \
        --distance-metric cosine --temperature 1.0 --novelty-threshold 0.35 --sub-centers 3
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import httpx

DETECTOR_FIELDS = [
    {"name": "ExtractedImages", "type": "array", "source": "postprocess:yolo_crops_base64"},
    {"name": "ProportionBoxes", "type": "array", "source": "postprocess:yolo_boxes_proportional"},
    {"name": "AbsoluteBoxes", "type": "array", "source": "postprocess:yolo_boxes_absolute"},
    {"name": "Confidences", "type": "array", "source": "postprocess:yolo_confidences"},
]
# Validator / SideClassifier / Classifier — ONNX output named "logits" + class_dict.
CLASSIFIER_FIELDS = [
    {"name": "Class", "type": "string", "source": "postprocess:softmax_max_label"},
    {"name": "Probability", "type": "float", "source": "postprocess:softmax_max"},
    {"name": "Probabilities", "type": "object", "source": "postprocess:softmax_dict"},
]
# Identifier (metric/NCC) — ONNX output "embedding" + an .npz sidecar (cls/centroids).
IDENTIFIER_FIELDS = [
    {"name": "Class", "type": "string", "source": "postprocess:ncc_argmax_label"},
    {"name": "Probability", "type": "float", "source": "postprocess:ncc_softmax_max"},
    {"name": "Probabilities", "type": "object", "source": "postprocess:ncc_softmax_dict"},
    # MUST be "float[]" (not "array"): the server's _assemble only flattens onnx-sourced
    # fields whose type ends in "[]" or starts with "float[" — anything else falls through to
    # value.item(), which raises on the multi-element embedding vector. (postprocess-sourced
    # fields above are assembled via their fn(ctx), so their "object"/"array" types are fine.)
    {"name": "Embedding", "type": "float[]", "source": "onnx:embedding"},
]
FIELDS_BY_TYPE = {
    "Detector": DETECTOR_FIELDS,
    "Validator": CLASSIFIER_FIELDS,
    "SideClassifier": CLASSIFIER_FIELDS,
    "Classifier": CLASSIFIER_FIELDS,
    "Identifier": IDENTIFIER_FIELDS,
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def export_to_onnx(weights: Path, img_size: int, out_dir: Path) -> Path:
    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("ultralytics not available — run with `uv run --with ultralytics --with httpx ...`")
    print(f"[export] {weights} -> ONNX (imgsz={img_size}, opset=17, simplify)")
    exported = Path(YOLO(str(weights)).export(format="onnx", imgsz=img_size, opset=17, simplify=True))
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"{weights.stem}.onnx"
    shutil.copyfile(exported, dest)
    print(f"[export] wrote {dest}")
    return dest


def stage(path: Path, out_dir: Path) -> Path:
    """Copy an already-prepared artifact into the source dir for file:// fetch."""
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / path.name
    if path.resolve() != dest.resolve():
        shutil.copyfile(path, dest)
    return dest


def load_class_dict(spec: str | None) -> dict[str, str] | None:
    """Accept a JSON file path or inline JSON. Normalize to {index-string: label}.
    Handles both {label: index} (toolkit class_dict) and {index: label}."""
    if not spec:
        return None
    raw = Path(spec).read_text() if Path(spec).exists() else spec
    d = json.loads(raw)
    # If values are ints (label -> index), invert to index -> label.
    if d and all(isinstance(v, int) for v in d.values()):
        d = {str(v): k for k, v in d.items()}
    return {str(k): str(v) for k, v in d.items()}


def build_card(args, onnx_path: Path, class_dict, classifier_file: Path | None) -> dict:
    entry = onnx_path.name
    node_type = args.node_type
    is_detector = node_type == "Detector"
    is_identifier = node_type == "Identifier"

    _hash = lambda p: sha256(p) if p.exists() else "0" * 64  # placeholder for --dry-run
    files = [{"name": entry, "url": onnx_path.absolute().as_uri(), "sha256": _hash(onnx_path)}]
    artifact = {"format": "onnx-bundle", "entrypoint": entry, "files": files}
    if classifier_file is not None:
        files.append({"name": classifier_file.name, "url": classifier_file.absolute().as_uri(),
                      "sha256": _hash(classifier_file)})
        artifact["classifier_file"] = classifier_file.name

    inference_config: dict = {"crop_format": "JPEG"}
    if is_detector:
        inference_config.update(conf_threshold=args.conf, iou_threshold=args.iou)
    if is_identifier:
        inference_config.update(distance_metric=args.distance_metric, temperature=args.temperature,
                                sub_centers=args.sub_centers)
        if args.novelty_threshold is not None:
            inference_config["novelty_threshold"] = args.novelty_threshold

    output: dict = {"contract": f"{node_type.lower()}-v1", "fields": FIELDS_BY_TYPE[node_type]}
    if class_dict is not None:
        output["class_dict"] = class_dict
    if is_identifier:
        output["produces_embedding"] = True

    return {
        "spec_version": "1.0.0",
        "job_id": f"local-{args.model_name}",
        "model_name": args.model_name,
        "population_id": args.population_id,
        "node_type": node_type,
        "input": {
            "contract": f"{node_type.lower()}-v1",
            "image_size": args.img_size,
            "channels": 3,
            # YOLO detectors: /255 only. Classifiers/identifiers: usually imagenet.
            "normalization": args.normalization or ("none" if is_detector else "imagenet"),
        },
        "output": output,
        "inference_config": inference_config,
        "artifact": artifact,
        "provenance": {"source": str(args.weights or args.onnx), "node_type": node_type},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-name", required=True, help="MUST equal the pipeline node's modelApi")
    ap.add_argument("--node-type", default="Detector",
                    choices=["Detector", "Validator", "SideClassifier", "Classifier", "Identifier"])
    ap.add_argument("--weights", type=Path, help="Ultralytics .pt (Detector only; auto-exported to ONNX)")
    ap.add_argument("--onnx", type=Path, help="Pre-exported .onnx (required for non-Detector types)")
    ap.add_argument("--img-size", type=int, default=640)
    ap.add_argument("--population-id", default=None)
    # detector
    ap.add_argument("--conf", type=float, default=0.15)
    ap.add_argument("--iou", type=float, default=0.5)
    # classifier / identifier
    ap.add_argument("--class-dict", default=None, help="JSON file/string mapping classes (index<->label)")
    ap.add_argument("--normalization", default=None, help="'none' | 'imagenet' (default by node type)")
    # identifier (NCC)
    ap.add_argument("--classifier-file", type=Path, default=None, help="NCC .npz sidecar (cls/centroids)")
    ap.add_argument("--distance-metric", default="cosine", choices=["cosine", "euclidean"])
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--novelty-threshold", type=float, default=None)
    ap.add_argument("--sub-centers", type=int, default=0)
    # plumbing
    ap.add_argument("--src-dir", type=Path, default=Path("/media/alex/Storage/finwave_models/_src"))
    ap.add_argument("--server", default=os.environ.get("FINWAVE_SERVER", "http://localhost:5003"))
    ap.add_argument("--api-key", default=os.environ.get("FINWAVE_API_KEY"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    # Validate per-type inputs.
    if args.node_type == "Detector":
        if not (args.weights or args.onnx):
            sys.exit("Detector needs --weights (.pt) or --onnx")
    else:
        if not args.onnx:
            sys.exit(f"{args.node_type} needs a pre-exported --onnx (export is model-specific; see the toolkit exporter)")
    if args.node_type in ("Validator", "SideClassifier", "Classifier", "Identifier") and not args.class_dict:
        sys.exit(f"{args.node_type} needs --class-dict")
    if args.node_type == "Identifier" and not args.classifier_file:
        sys.exit("Identifier needs --classifier-file (the NCC .npz)")

    # Resolve the ONNX artifact.
    if args.dry_run:
        onnx_path = args.onnx or (args.weights.with_suffix(".onnx") if args.weights else Path("model.onnx"))
    elif args.onnx:
        onnx_path = stage(args.onnx, args.src_dir)
    elif args.weights.suffix == ".onnx":
        onnx_path = stage(args.weights, args.src_dir)
    else:
        onnx_path = export_to_onnx(args.weights, args.img_size, args.src_dir)

    classifier_file = None
    if args.classifier_file is not None:
        classifier_file = args.classifier_file if args.dry_run else stage(args.classifier_file, args.src_dir)

    card = build_card(args, onnx_path, load_class_dict(args.class_dict), classifier_file)
    print(json.dumps(card, indent=2))
    if args.dry_run:
        print("\n[dry-run] card not registered")
        return

    if not args.api_key:
        sys.exit("no API key: pass --api-key or set FINWAVE_API_KEY")
    url = f"{args.server.rstrip('/')}/models/register"
    print(f"\n[register] POST {url}  (model_name={args.model_name}, node_type={args.node_type})")
    resp = httpx.post(url, json={"model_card": card}, headers={"X-API-KEY": args.api_key}, timeout=300.0)
    if resp.status_code >= 400:
        sys.exit(f"[register] FAILED {resp.status_code}: {resp.text}")
    print(f"[register] OK: {resp.json()}")


if __name__ == "__main__":
    main()
