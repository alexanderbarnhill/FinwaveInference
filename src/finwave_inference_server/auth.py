"""API key dependency for management endpoints.

Inference is unauthenticated (network-trusted), matching the existing
.NET 6 connector's pattern. Only `/models/*` endpoints require the key.
"""
from fastapi import Depends, Header, HTTPException, status

from .config import Settings, get_settings


async def api_key_required(
    x_api_key: str | None = Header(default=None),
    settings: Settings = Depends(get_settings),
) -> None:
    if x_api_key is None or x_api_key != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required",
        )
