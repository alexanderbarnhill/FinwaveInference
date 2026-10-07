"""Manifest warmup: URL resolution, shape normalization, and the register loop
(fetching and sha256 verification are covered by registry/loader tests; here we
stub the HTTP client and the registry so no ONNX or network is needed)."""
from __future__ import annotations

import json
import os

os.environ.setdefault("FINWAVE_API_KEY", "test-key")

import pytest

from finwave_inference_server import warmup
from finwave_inference_server.config import Settings
from finwave_inference_server.schemas import FinwaveModelCard


def _card(name: str, onnx_url: str = "model.onnx", sha: str = "a" * 64) -> dict:
    return {
        "spec_version": "1.0",
        "job_id": "job-1",
        "model_name": name,
        "node_type": "Detector",
        "input": {"contract": "image/v1", "image_size": 640},
        "output": {"contract": "detections/v1", "fields": []},
        "artifact": {
            "format": "onnx-bundle",
            "entrypoint": "model.onnx",
            "files": [{"name": "model.onnx", "url": onnx_url, "sha256": sha}],
        },
    }


# --- pure helpers -----------------------------------------------------------

def test_resolve_absolute_passes_through() -> None:
    assert warmup._resolve("https://x/models/m.json?sig=S", "file:///abs/a.onnx") == "file:///abs/a.onnx"
    assert warmup._resolve("https://x/models/m.json?sig=S", "https://y/z.onnx") == "https://y/z.onnx"


def test_resolve_relative_inherits_path_and_sas() -> None:
    base = "https://acct.blob.core.windows.net/models/manifest.json?sig=SAS&se=2026"
    assert (
        warmup._resolve(base, "FIN_DETECT/card.json")
        == "https://acct.blob.core.windows.net/models/FIN_DETECT/card.json?sig=SAS&se=2026"
    )


def test_resolve_relative_keeps_own_query_when_present() -> None:
    base = "https://acct/models/m.json?sig=SAS"
    assert warmup._resolve(base, "a/b.onnx?sig=OTHER") == "https://acct/models/a/b.onnx?sig=OTHER"


@pytest.mark.parametrize(
    "doc,n",
    [
        ({"models": [{"card_url": "a"}, {"card_url": "b"}]}, 2),
        ({"Models": [{"card_url": "a"}]}, 1),
        ([{"card_url": "a"}, "junk", {"card_url": "b"}], 2),
        ({"nope": 1}, 0),
        ("garbage", 0),
    ],
)
def test_normalize_entries(doc: object, n: int) -> None:
    assert len(warmup._normalize_entries(doc)) == n


def test_rebase_card_resolves_artifact_urls() -> None:
    card = FinwaveModelCard.model_validate(_card("FIN_DETECT", onnx_url="model.onnx"))
    base = "https://acct/models/FIN_DETECT/card.json?sig=SAS"
    rebased = warmup._rebase_card(card, base)
    assert rebased.artifact.files[0].url == "https://acct/models/FIN_DETECT/model.onnx?sig=SAS"


# --- warm_from_manifest with stubbed client + registry ----------------------

class _Resp:
    def __init__(self, payload: object) -> None:
        self._payload = payload
        self.text = json.dumps(payload)

    def raise_for_status(self) -> None:  # noqa: D401
        return None

    def json(self) -> object:
        return self._payload


class _FakeClient:
    """Serves canned responses by URL path suffix; records GETs."""

    def __init__(self, routes: dict[str, object]) -> None:
        self._routes = routes
        self.gets: list[str] = []

    async def __aenter__(self) -> "_FakeClient":
        return self

    async def __aexit__(self, *_a: object) -> None:
        return None

    async def get(self, url: str) -> _Resp:
        self.gets.append(url)
        for suffix, payload in self._routes.items():
            if url.split("?", 1)[0].endswith(suffix):
                return _Resp(payload)
        raise AssertionError(f"no route for {url}")


class _FakeRegistry:
    def __init__(self, preloaded: dict[str, FinwaveModelCard] | None = None) -> None:
        self._loaded = preloaded or {}
        self.registered: list[str] = []

    def loaded_names(self) -> set[str]:
        return set(self._loaded)

    def loaded_card(self, name: str) -> FinwaveModelCard | None:
        return self._loaded.get(name)

    async def register(self, card: FinwaveModelCard) -> None:
        self.registered.append(card.model_name)
        self._loaded[card.model_name] = card


def _settings(url: str | None) -> Settings:
    return Settings(api_key="k", model_manifest_url=url)


async def test_warmup_noop_when_unset() -> None:
    reg = _FakeRegistry()
    res = await warmup.warm_from_manifest(reg, _settings(None))
    assert (res.registered, res.skipped, res.failed) == (0, 0, 0)
    assert reg.registered == []


async def test_warmup_registers_card_url_models(monkeypatch) -> None:
    manifest = {"models": [{"card_url": "FIN_DETECT/card.json"}, {"card_url": "WAKW_miew_id/card.json"}]}
    routes = {
        "manifest.json": manifest,
        "FIN_DETECT/card.json": _card("FIN_DETECT"),
        "WAKW_miew_id/card.json": _card("WAKW_miew_id"),
    }
    client = _FakeClient(routes)
    monkeypatch.setattr(warmup.httpx, "AsyncClient", lambda *a, **k: client)
    reg = _FakeRegistry()

    res = await warmup.warm_from_manifest(reg, _settings("https://acct/models/manifest.json?sig=SAS"))

    assert res.registered == 2 and res.failed == 0
    assert set(reg.registered) == {"FIN_DETECT", "WAKW_miew_id"}
    # the card.json fetch inherited the manifest SAS
    assert any("FIN_DETECT/card.json?sig=SAS" in g for g in client.gets)


async def test_warmup_skips_already_current_and_survives_bad_entry(monkeypatch) -> None:
    preloaded = {"FIN_DETECT": FinwaveModelCard.model_validate(_card("FIN_DETECT", sha="a" * 64))}
    manifest = {"models": [
        {"card_url": "FIN_DETECT/card.json"},   # same sha -> skip
        {"card_url": "BROKEN/card.json"},        # 404-ish -> failed, not fatal
        {"card": _card("NEW_MODEL")},            # inline -> registered
    ]}
    routes = {
        "manifest.json": manifest,
        "FIN_DETECT/card.json": _card("FIN_DETECT", sha="a" * 64),
    }
    client = _FakeClient(routes)
    monkeypatch.setattr(warmup.httpx, "AsyncClient", lambda *a, **k: client)
    reg = _FakeRegistry(preloaded)

    res = await warmup.warm_from_manifest(reg, _settings("https://acct/models/manifest.json?sig=SAS"))

    assert res.skipped == 1
    assert res.failed == 1
    assert reg.registered == ["NEW_MODEL"]
