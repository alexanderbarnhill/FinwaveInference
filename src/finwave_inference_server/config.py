"""Env-driven settings.

All variables are prefixed `FINWAVE_`. `.env` is read in development; in
containers prefer real env vars (set via the orchestrator's secret store).
"""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="FINWAVE_",
        env_file=".env",
        extra="ignore",
    )

    api_key: str
    model_store_path: str = "/var/lib/finwave/models"
    model_manifest_url: str | None = None
    supported_spec_major: int = 1
    listen_host: str = "0.0.0.0"
    listen_port: int = 5003


@lru_cache
def get_settings() -> Settings:
    return Settings()
