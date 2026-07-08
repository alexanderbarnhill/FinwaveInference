#!/usr/bin/env python3
"""Export a trained Ultralytics YOLO detector to ONNX and register it with a
running FinWave inference server, in one step.

This is the bring-up helper for local hosting: point it at a `.pt` (or an
already-exported `.onnx`), give it the model name that MUST equal the pipeline
node's `modelApi`, and it will:

  1. export .pt -> .onnx at the requested image size (skipped if given .onnx),
  2. copy the .onnx into a source dir and compute its sha256,
  3. build a FinwaveModelCard (Detector, YOLO postprocess fields),
  4. POST it to <server>/models/register (X-API-KEY),

so the server loads it immediately and re-loads it on every restart.

Examples
--------
Dorsal fin (weights already local), served at the node's modelApi name:

    uv run --with ultralytics --with httpx python scripts/export_and_register.py \
      --weights /home/alex/git/finwave/toolkit/fin_detect/detect_data/runs/detect/runs/eadd-b-full/weights/best.pt \
      --model-name fin-detector \
      --img-size 640

Eye patch (weights fetched from the NAS), re-exported small for a weak GPU:

    uv run --with ultralytics --with httpx python scripts/export_and_register.py \
      --weights ./ep_best.pt --model-name eye-patch-detector --img-size 640

Write the card only (no export, no POST) to inspect it first:

    ... --weights model.onnx --model-name fin-detector --dry-run
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
    # Names MUST match the worker's canonical detector normalizer + DetectorHandler
    # (workers/finwave-inference-worker: contracts/normalize.py, node_handlers.py).
    {"name": "ExtractedImages", "type": "array", "source": "postprocess:yolo_crops_base64"},
    {"name": "ProportionBoxes", "type": "array", "source": "postprocess:yolo_boxes_proportional"},
    {"name": "AbsoluteBoxes", "type": "array", "source": "postprocess:yolo_boxes_absolute"},
    {"name": "Confidences", "type": "array", "source": "postprocess:yolo_confidences"},
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def export_to_onnx(weights: Path, img_size: int, out_dir: Path) -> Path:
    """Export an Ultralytics .pt to .onnx at img_size. Returns the .onnx path."""
    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit(
            "ultralytics not available — run with:\n"
            "  uv run --with ultralytics --with httpx python scripts/export_and_register.py ..."
        )
    print(f"[export] {weights} -> ONNX (imgsz={img_size}, opset=17, simplify)")
    model = YOLO(str(weights))
    exported = Path(model.export(format="onnx", imgsz=img_size, opset=17, simplify=True))
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"{weights.stem}.onnx"
    shutil.copyfile(exported, dest)
    print(f"[export] wrote {dest}")
    return dest


def build_card(args, onnx_path: Path) -> dict:
    entry = onnx_path.name
    return {
        "spec_version": "1.0.0",
        "job_id": f"local-{args.model_name}",
        "model_name": args.model_name,
        "population_id": args.population_id,
        "node_type": args.node_type,
        "input": {
            "contract": "detector-v1",
            "image_size": args.img_size,
            "channels": 3,
            # Ultralytics/YOLO models expect only /255 scaling (done by the server);
            # do NOT apply imagenet mean/std or detections will be garbage.
            "normalization": "none",
        },
        "output": {
            "contract": "detector-v1",
            "fields": DETECTOR_FIELDS,
            "probabilistic": True,
        },
        "inference_config": {
            "conf_threshold": args.conf,
            "iou_threshold": args.iou,
            "crop_format": "JPEG",
        },
        "artifact": {
            "format": "onnx-bundle",
            "entrypoint": entry,
            "files": [
                {
                    "name": entry,
                    "url": onnx_path.absolute().as_uri(),
                    "sha256": sha256(onnx_path),
                }
            ],
        },
        "provenance": {"source_weights": str(args.weights), "exported_img_size": args.img_size},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True, type=Path, help="Path to trained .pt (exported) or an .onnx (used as-is)")
    ap.add_argument("--model-name", required=True, help="MUST equal the pipeline node's modelApi (e.g. fin-detector)")
    ap.add_argument("--img-size", type=int, default=640, help="Export/inference square size (default 640; use smaller for a weak GPU)")
    ap.add_argument("--node-type", default="Detector", help="Card node_type (default Detector)")
    ap.add_argument("--population-id", default=None, help="Optional population id to stamp on the card")
    ap.add_argument("--conf", type=float, default=0.15, help="Confidence threshold (default 0.15)")
    ap.add_argument("--iou", type=float, default=0.5, help="NMS IoU threshold (default 0.5)")
    ap.add_argument("--src-dir", type=Path, default=Path("/media/alex/Storage/finwave_models/_src"), help="Where the .onnx is staged for the server to fetch via file://")
    ap.add_argument("--server", default=os.environ.get("FINWAVE_SERVER", "http://localhost:5003"), help="Inference server base URL")
    ap.add_argument("--api-key", default=os.environ.get("FINWAVE_API_KEY"), help="Server API key (or set FINWAVE_API_KEY)")
    ap.add_argument("--dry-run", action="store_true", help="Build + print the card; do not export or POST")
    args = ap.parse_args()

    # 1/2. Resolve the ONNX artifact (export if we were given a .pt).
    if args.dry_run:
        onnx_path = args.weights if args.weights.suffix == ".onnx" else args.weights.with_suffix(".onnx")
    elif args.weights.suffix == ".onnx":
        args.src_dir.mkdir(parents=True, exist_ok=True)
        onnx_path = args.src_dir / args.weights.name
        if args.weights.resolve() != onnx_path.resolve():
            shutil.copyfile(args.weights, onnx_path)
    else:
        onnx_path = export_to_onnx(args.weights, args.img_size, args.src_dir)

    # 3. Build the card.
    card = build_card(args, onnx_path)
    print(json.dumps(card, indent=2))
    if args.dry_run:
        print("\n[dry-run] card not registered")
        return

    if not args.api_key:
        sys.exit("no API key: pass --api-key or set FINWAVE_API_KEY (must match the server's FINWAVE_API_KEY)")

    # 4. Register.
    url = f"{args.server.rstrip('/')}/models/register"
    print(f"\n[register] POST {url}  (model_name={args.model_name})")
    resp = httpx.post(url, json={"model_card": card}, headers={"X-API-KEY": args.api_key}, timeout=300.0)
    if resp.status_code >= 400:
        sys.exit(f"[register] FAILED {resp.status_code}: {resp.text}")
    print(f"[register] OK: {resp.json()}")
    print(f"[register] verify:  curl -s -H 'X-API-KEY: ***' {args.server.rstrip('/')}/models")


if __name__ == "__main__":
    main()
