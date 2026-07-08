# Local hosting — bring-up runbook

Run finwave's ML inference on a local machine (behind a residential/changing IP)
and connect it to finwave in Azure. No inbound connectivity is required: a
co-located **pull worker** claims jobs from Azure Service Bus, calls this server
over `localhost`, and posts results back — all outbound-only.

```
Azure (Hub)  ──enqueue──►  Service Bus: local-q-inference-requests
                                              │  (worker pulls, outbound only)
   LOCAL:  finwave-inference-worker ──localhost──► finwave-inference-server (this repo)
                                              │
           results ──► Service Bus: prod-q-inference-results ──► Hub result consumer
```

## Prerequisites (this machine)

- Linux + systemd, `uv`, an NVIDIA GPU is a bonus (an 8 GB card runs yolov8n/s
  easily; CPU also works for small models).
- This repo's venv: `cd inference-server && uv sync`.
- The worker repo checked out with its venv: `cd workers/finwave-inference-worker && uv venv && uv pip install -r requirements.txt`.
- Model artifacts staged under `/media/alex/Storage/finwave_models/` (`.pt` or `.onnx`).

## 1. Inference server

```bash
sudo install -d -o $USER /media/alex/Storage/finwave_models/_store
sudo install -D deploy/inference-server.env.example /etc/finwave/inference-server.env
sudo chmod 600 /etc/finwave/inference-server.env   # then edit: set FINWAVE_API_KEY
sudo cp deploy/finwave-inference-server.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now finwave-inference-server
curl -s localhost:5003/health    # {"status":"ok","models_loaded":0}
```

Optional GPU: `uv pip install onnxruntime-gpu` in the server venv (it auto-selects
CUDA when available) and add `SupplementaryGroups=video render` to the unit.

## 2. Register the detectors

One command per model — export to ONNX + build the card + register. The
`--model-name` **must equal the pipeline node's `modelApi`** for that population
(see step 4).

```bash
# Dorsal fin
uv run --with ultralytics --with httpx python scripts/export_and_register.py \
  --weights /media/alex/Storage/finwave_models/<fin>/best.pt \
  --model-name <fin-detector-modelApi> --img-size 640 \
  --api-key "$FINWAVE_API_KEY"

# Eye patch
uv run --with ultralytics --with httpx python scripts/export_and_register.py \
  --weights /media/alex/Storage/finwave_models/<eye>/best.pt \
  --model-name <eye-patch-modelApi> --img-size 640 \
  --api-key "$FINWAVE_API_KEY"

curl -s -H "X-API-KEY: $FINWAVE_API_KEY" localhost:5003/models   # 2 models
```

Registered cards persist under the store dir and reload automatically on restart.

## 3. Pull worker

```bash
sudo install -D deploy/local-worker.env.example /etc/finwave/inference-worker.env  # (from the worker repo)
sudo chmod 600 /etc/finwave/inference-worker.env    # fill in SB_CONNECTION, BLOB_CONNECTION
sudo cp deploy/finwave-inference-worker.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now finwave-inference-worker
journalctl -u finwave-inference-worker -f
```

Smoke-test one job without touching Azure:
`.venv/bin/python src/main.py --once <job.json>` (uses `ImageBase64` inline or Blob).

## 4. Hub routing (Azure — deploy required)

The Hub must send the target population's inference jobs to `local-q-inference-requests`
and set the per-job `InferenceBaseUrl` to `http://localhost:5003`. See the
companion change in `app/` (OutboxDispatcher + `InferenceService:LocalPopulations`
config). A DAG PipelineDefinition must exist for the population with Detector
nodes whose `modelApi` matches the registered model names.

## Who does what

- **Code (in these repos):** the `/api/inference` alias, per-box confidence, the
  export/register helper, these unit files, and the Hub routing change.
- **You (Azure ops):** deploy the Hub change; mint a queue-scoped Service Bus SAS
  and a container-scoped Blob SAS; set `LocalPopulations`.
- **You (local):** stage model weights, fill the env files, `systemctl enable`.
