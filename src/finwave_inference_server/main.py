"""FastAPI app for the FinWave inference server."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse

from .auth import api_key_required
from .config import get_settings
from .registry import ModelRegistry
from .schemas import InferenceRequest, RegisterModelRequest

log = logging.getLogger("finwave.inference")

_registry: ModelRegistry | None = None


def get_registry() -> ModelRegistry:
    if _registry is None:
        raise RuntimeError("registry not initialised; called outside the app lifespan?")
    return _registry


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _registry
    settings = get_settings()
    _registry = ModelRegistry(settings)
    # Reload models persisted in the store so a restart doesn't drop everything
    # and require manual re-registration.
    reloaded = _registry.reload_persisted()
    log.info(
        "registry initialised; model_store=%s; reloaded %d persisted model(s)",
        settings.model_store_path, reloaded,
    )
    yield
    _registry = None


app = FastAPI(title="FinWave Inference Server", lifespan=lifespan)


def _ensure_image_within_limit(image_b64: str) -> None:
    """Reject oversized images (base64 length → decoded bytes, no full decode)."""
    approx_bytes = (len(image_b64) * 3) // 4
    limit_mb = get_settings().max_image_mb
    if approx_bytes > limit_mb * 1024 * 1024:
        raise HTTPException(413, detail=f"image exceeds the {limit_mb} MB limit")  # 413 Content Too Large


@app.middleware("http")
async def _limit_body_size(request: Request, call_next):
    # Cheap early guard: reject obviously-oversized bodies before reading/parsing them. The
    # precise per-image check is in /inference; this caps the base64 image (~+33%) + JSON wrapper.
    content_length = request.headers.get("content-length")
    if content_length and content_length.isdigit():
        if int(content_length) > get_settings().max_image_mb * 1024 * 1024 * 2:
            return JSONResponse(status_code=413, content={"detail": "request body too large"})
    return await call_next(request)


@app.get("/health")
async def health(registry: ModelRegistry = Depends(get_registry)) -> dict:
    return {"status": "ok", "models_loaded": len(registry)}


@app.get("/ready")
async def ready() -> dict:
    # Readiness probe: the server can serve inference only once ≥1 model is loaded. Distinct
    # from /health (200 whenever the process is up) so an orchestrator doesn't route inference
    # traffic to a cold instance that would 404 every request. 503 until warm.
    if _registry is None or len(_registry) == 0:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail="no models loaded")
    return {"status": "ready", "models_loaded": len(_registry)}


@app.post("/inference")
async def inference(
    req: InferenceRequest,
    registry: ModelRegistry = Depends(get_registry),
) -> dict:
    _ensure_image_within_limit(req.image)
    try:
        return await registry.run(req.model_name, req.image)
    except KeyError as e:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=str(e))


@app.post("/models/register", dependencies=[Depends(api_key_required)])
async def register_model(
    req: RegisterModelRequest,
    registry: ModelRegistry = Depends(get_registry),
) -> dict:
    try:
        await registry.register(req.model_card)
    except ValueError as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=str(e))
    except KeyError as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=f"unknown postprocess: {e}")
    return {"status": "registered", "model_name": req.model_card.model_name}


@app.get("/models", dependencies=[Depends(api_key_required)])
async def list_models(registry: ModelRegistry = Depends(get_registry)) -> list[dict]:
    return registry.list()
