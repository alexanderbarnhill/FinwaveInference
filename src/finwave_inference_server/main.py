"""FastAPI app for the FinWave inference server."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, status

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
    if settings.model_manifest_url:
        log.warning(
            "model_manifest_url=%s set, but warmup loading is not implemented yet",
            settings.model_manifest_url,
        )
    log.info("registry initialised; model_store=%s", settings.model_store_path)
    yield
    _registry = None


app = FastAPI(title="FinWave Inference Server", lifespan=lifespan)


@app.get("/health")
async def health(registry: ModelRegistry = Depends(get_registry)) -> dict:
    return {"status": "ok", "models_loaded": len(registry)}


@app.post("/inference")
async def inference(
    req: InferenceRequest,
    registry: ModelRegistry = Depends(get_registry),
) -> dict:
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
