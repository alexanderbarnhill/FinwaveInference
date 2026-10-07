"""Test isolation for the env-driven, lru-cached settings.

`get_settings()` is `@lru_cache`d, and several tests redirect `FINWAVE_*`
(notably `FINWAVE_MODEL_STORE_PATH`) to point the registry at a tmp store. Set
once and left cached, that leaks into later tests: a test that expects an empty
registry (e.g. /ready = 503) instead reloads the previous test's persisted
model. This autouse fixture clears the settings cache around every test and
strips the store/manifest env so each test starts from a clean, default config
regardless of order.
"""
from __future__ import annotations

import pytest

from finwave_inference_server.config import get_settings


@pytest.fixture(autouse=True)
def _isolate_settings(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("FINWAVE_MODEL_STORE_PATH", raising=False)
    monkeypatch.delenv("FINWAVE_MODEL_MANIFEST_URL", raising=False)
    get_settings.cache_clear()  # type: ignore[attr-defined]
    yield
    get_settings.cache_clear()  # type: ignore[attr-defined]
