# finwave-inference-server

FastAPI + onnxruntime service that replaces the `FinwaveInferenceConnector` + TorchServe pair. Consumes a `FinwaveModelCard` (see `app/docs/architecture/contracts/backend-ml.md`) and serves ONNX models declared by it.

## Status

**Skeleton.** Wired end-to-end for the Identifier paradigm (ONNX encoder + NCC classifier sidecar). YOLO Detector postprocess is stubbed (`NotImplementedError`); MAR adapter for legacy models is not implemented. Warmup-from-manifest is implemented (`FINWAVE_MODEL_MANIFEST_URL`), so a fresh box can self-load its assigned models at startup. See `backend-ml.md` for the migration plan.

## What it does

| Method | Path | Auth | Purpose |
|---|---|---|---|
| `GET` | `/health` | none | Liveness + count of loaded models. |
| `POST` | `/inference` | none (network-trusted) | Run a model. Body: `{"model_name": "...", "image": "<base64>"}`. Returns the dict declared by the model card's `output.fields[]`. |
| `POST` | `/models/register` | `X-API-KEY` | Body: `{"model_card": {...}}`. Fetches artifact files by URL, verifies sha256s, loads into the in-memory registry. |
| `GET` | `/models` | `X-API-KEY` | List loaded models. |

The inference endpoint deliberately matches the unauthenticated network-trust pattern of the existing .NET 6 connector — access control is at the network layer, not the app layer.

## Stack

- Python 3.11+
- FastAPI / Uvicorn
- onnxruntime (CPU). For GPU, replace `onnxruntime` with `onnxruntime-gpu` at install time (in the Dockerfile add a build arg or use a separate base image).
- Pydantic v2 for the runtime subset of `FinwaveModelCard`. When the shared `finwave-contracts` Python package exists, `src/finwave_inference_server/schemas.py` is deleted and imports come from there.

## Layout

```
inference-server/
├── pyproject.toml
├── Dockerfile
├── README.md
├── src/
│   └── finwave_inference_server/
│       ├── main.py          # FastAPI app + endpoints
│       ├── config.py        # env-driven Settings
│       ├── auth.py          # X-API-KEY dependency
│       ├── schemas.py       # ModelCard subset (will move to finwave-contracts)
│       ├── loader.py        # fetch + sha256 verify + onnxruntime session creation
│       ├── registry.py      # ModelRegistry (in-memory) + preprocess + dispatch
│       └── postprocess/
│           ├── _registry.py # decorator-based fn registry
│           ├── ncc.py       # NCC argmax/softmax/novelty for Identifier
│           ├── passthrough.py  # softmax_max{,_label,_dict} + quality_score
│           └── yolo.py      # stubs (port pending)
└── tests/
    └── test_health.py
```

## Configuration

All variables are prefixed `FINWAVE_`. In dev a `.env` file in this directory is loaded; in containers set real environment variables.

| Variable | Default | Purpose |
|---|---|---|
| `FINWAVE_API_KEY` | — (required) | Shared secret for `/models/*` endpoints. |
| `FINWAVE_MODEL_STORE_PATH` | `/var/lib/finwave/models` | Where downloaded artifact files are cached. |
| `FINWAVE_MODEL_MANIFEST_URL` | unset | Manifest of ModelCards to preload on startup (see "Manifest warmup"). A relative `card_url` / artifact URL inside it resolves against the manifest URL and inherits its query string, so one SAS on the manifest URL covers every nested fetch. |
| `FINWAVE_SUPPORTED_SPEC_MAJOR` | `1` | Reject cards with a different major version. |
| `FINWAVE_LISTEN_HOST` | `0.0.0.0` | |
| `FINWAVE_LISTEN_PORT` | `5003` | |

## Running locally

```bash
cd inference-server
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
echo "FINWAVE_API_KEY=dev-key" > .env
uvicorn finwave_inference_server.main:app --reload --port 5003
```

```bash
curl http://localhost:5003/health
# {"status":"ok","models_loaded":0}

curl -H "X-API-KEY: dev-key" http://localhost:5003/models
# []
```

Run the tests:

```bash
pytest
```

## Registering a model

POST a `FinwaveModelCard` to `/models/register`. The card's `artifact.files[].url` fields must be reachable from this service. Example minimal Identifier card (abbreviated):

```jsonc
{
  "model_card": {
    "spec_version": "1.0.0",
    "job_id": "9b2f1c34-...",
    "model_name": "WAKW_dinov2_vits14_v9",
    "population_id": "WAKW",
    "node_type": "Identifier",
    "input":  { "contract": "Detection", "image_size": 384, "channels": 3, "normalization": "imagenet" },
    "output": {
      "contract": "IdentificationResult",
      "fields": [
        {"name":"Class",         "type":"string",            "source":"postprocess:ncc_argmax_label"},
        {"name":"Probability",   "type":"float",             "source":"postprocess:ncc_softmax_max"},
        {"name":"Probabilities", "type":"map<string,float>", "source":"postprocess:ncc_softmax_dict"},
        {"name":"Embedding",     "type":"float[]", "dim":512, "source":"onnx:embedding"}
      ],
      "produces_embedding": true,
      "embedding_dim": 512,
      "probabilistic": true,
      "class_dict": {"0":"WAKW-001", "1":"WAKW-002"}
    },
    "inference_config": {
      "distance_metric": "cosine",
      "temperature": 0.0234,
      "novelty_threshold": 0.412,
      "sub_centers": 3
    },
    "artifact": {
      "format": "onnx-bundle",
      "entrypoint": "encoder.onnx",
      "classifier_file": "classifier.npz",
      "files": [
        {"name":"encoder.onnx",   "url":"https://trainer/.../encoder.onnx",   "sha256":"..."},
        {"name":"classifier.npz", "url":"https://trainer/.../classifier.npz", "sha256":"..."}
      ]
    }
  }
}
```

The `classifier.npz` for the NCC paradigm contains: `cls` (class IDs), `centroids` (and optionally `sub_center_weights`), produced by `pipeline_ml/deploy.py:build_mar` today and adapted to ship as a sidecar instead of being embedded in the MAR.

## Manifest warmup

Set `FINWAVE_MODEL_MANIFEST_URL` and the server registers every model the manifest lists at startup (after reloading anything already in the store), so a fresh box, for example a newly deployed worker runner, comes up with its assigned models loaded instead of serving `503` from `/ready` until each one is POSTed by hand. Registration reuses the same fetch + sha256 verification + local persistence as `/models/register`, so after the first warmup the artifacts live in the store and a later restart reloads them with no network and no dependence on a by-then-expired SAS.

The manifest is JSON. Any of these shapes is accepted:

```jsonc
{"models": [ {"card_url": "FIN_DETECT/card.json"}, {"card_url": "WAKW_miew_id/card.json"} ]}
{"models": [ {"card": { /* ...FinwaveModelCard... */ }} ]}   // inline cards
[ { /* ...FinwaveModelCard... */ } ]                         // bare list of cards
```

A relative `card_url` (and a relative `artifact.files[].url` inside a fetched card) resolves against its parent document's URL and inherits that URL's query string. So the Hub can mint one read SAS, append it to the manifest URL, and lay the blob out as:

```
models/manifest.json          <- FINWAVE_MODEL_MANIFEST_URL (?<sas>)
models/FIN_DETECT/card.json   <- card_url "FIN_DETECT/card.json"
models/FIN_DETECT/model.onnx  <- the card's artifact file "model.onnx"
```

and every nested fetch carries the same SAS with no per-URL rewriting. Absolute URLs (any scheme, including `file://`) pass through untouched. Warmup is best-effort: a failure on the manifest fetch or any single model is logged and never blocks startup; the server still serves whatever reloaded from the store and still accepts `/models/register`. A model already loaded with a matching entrypoint `sha256` is skipped; a changed `sha256` re-registers (so bumping a model in the manifest updates the box on its next restart).

## Building the container

```bash
docker build -t finwave-inference-server:dev .
docker run --rm -p 5003:5003 \
  -e FINWAVE_API_KEY=dev-key \
  -v /tmp/finwave-models:/var/lib/finwave/models \
  finwave-inference-server:dev
```

If you're behind a VPN that black-holes the default Docker bridge network (eduVPN does this), add `--network=host` to the build:

```bash
docker build --network=host -t finwave-inference-server:dev .
```

CI doesn't need this — `.github/workflows/docker.yml` builds and pushes to GHCR on every push to `main`, on tags `v*`, and on manual dispatch. Pull from `ghcr.io/alexanderbarnhill/finwaveinference` once it's run.

## Where this lives in the workspace

`inference-server/` sits next to the existing `inference-connector/` (.NET 6) during the transition. The two services coexist: TorchServe + the .NET 6 connector keep serving until each model is migrated to a `FinwaveModelCard` and registered here. See `app/docs/architecture/contracts/backend-ml.md` for the per-model cutover plan.

When migration completes, the `inference-connector` repo can be archived. Whether `inference-server/` becomes its own repo, replaces the `inference-connector` repo, or stays workspace-local is an open question — pick what fits ops.

## Not yet implemented

- **MAR adapter** — `artifact.format == "mar"` is currently rejected. The legacy adapter exists in the spec for the transition window but hasn't been written.
- **`shutdown` semantics** — sessions are dropped on process exit; no explicit `unregister` endpoint yet.
- **Metrics** — `/health` is the only observability hook. Prometheus exposition belongs here once cutover starts.

## Implemented postprocess library

Functions registered by name in `ModelCard.output.fields[].source`:

| Name | Used by | Behaviour |
|---|---|---|
| `ncc_argmax_label` | Identifier, Classifier | Min-distance class label; respects `sub_centers`. |
| `ncc_softmax_max` | Identifier | Max softmax probability with calibrated `temperature`. |
| `ncc_softmax_dict` | Identifier | Full `{class: probability}` mapping. |
| `is_novel` | Identifier | `min_distance > novelty_threshold`. |
| `softmax_max_label` | SideClassifier, Validator | Argmax-of-softmax over an ONNX `logits` output. |
| `softmax_max` | SideClassifier, Validator | Max softmax probability. |
| `softmax_dict` | SideClassifier, Validator | Full `{class: probability}` mapping. |
| `quality_score` | QualityScorer | Normalize the MLP output using saved `mean`/`std` from the sidecar. |
| `yolo_boxes_proportional` | Detector | NMS + bbox decode + letterbox-undo → `[{X,Y,W,H}]` in [0,1] (centre-based). |
| `yolo_boxes_absolute` | Detector | Same, in pixel coords. |
| `yolo_crops_base64` | Detector | Crops the original image at each kept box and returns base64 JPEG/PNG (`crop_format` in `inference_config`). |
| `yolo_confidences` | Detector | `[float]` of kept-box confidence scores. |

YOLO thresholds come from `inference_config.conf_threshold` (default 0.15) and `inference_config.iou_threshold` (default 0.5).
