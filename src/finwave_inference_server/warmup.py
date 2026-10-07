"""Warm the registry from a manifest of ModelCards on startup.

When `FINWAVE_MODEL_MANIFEST_URL` is set, the server fetches that manifest at
startup (after reloading anything already persisted in the store) and registers
every model it lists. This is what lets a fresh box, for example a newly
deployed worker runner, come up with its assigned models already loaded instead
of serving 503 until someone manually POSTs each one to `/models/register`.

Manifest shape (JSON). Any of these is accepted, so the Hub side can emit
whichever is convenient:

    {"models": [ {"card_url": "FIN_DETECT/card.json"}, ... ]}   # URLs to cards
    {"models": [ {"card": { ...FinwaveModelCard... }}, ... ]}    # inline cards
    {"models": [ { ...FinwaveModelCard... }, ... ]}              # bare cards
    [ ... ]                                                       # bare list

URL resolution. A relative `card_url` (and a relative `artifact.files[].url`
inside a card) is resolved against its parent document's URL, and inherits that
URL's query string. So the Hub can mint ONE read SAS, append it to the manifest
URL, and lay the blob out as:

    models/manifest.json              <- FINWAVE_MODEL_MANIFEST_URL (?<sas>)
    models/FIN_DETECT/card.json       <- card_url "FIN_DETECT/card.json"
    models/FIN_DETECT/model.onnx      <- card's artifact file "model.onnx"

and every nested fetch carries the same SAS without the Hub rewriting each URL.
Absolute URLs (anything with a scheme, including `file://`) pass through
untouched, so a card may also point at artifacts hosted elsewhere.

Fetching and verification reuse `ModelRegistry.register` unchanged: each
artifact's sha256 is checked, and the card is persisted locally (its URLs
rewritten to `file://`) so a later restart reloads it with no network and no
dependence on a by-then-expired SAS.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from urllib.parse import urljoin, urlsplit, urlunsplit

import httpx

from .config import Settings
from .registry import ModelRegistry
from .schemas import FinwaveModelCard

log = logging.getLogger("finwave.inference")


@dataclass
class WarmupResult:
    registered: int = 0
    skipped: int = 0  # already loaded with a matching artifact
    failed: int = 0


def _inherit_query(url: str, base: str) -> str:
    """If `url` has no query of its own, carry over `base`'s query (the SAS)."""
    parts = urlsplit(url)
    if parts.query:
        return url
    base_query = urlsplit(base).query
    if not base_query:
        return url
    return urlunsplit(parts._replace(query=base_query))


def _resolve(base: str, ref: str) -> str:
    """Resolve `ref` against `base`. Absolute refs (with a scheme) pass through;
    relative refs resolve against the base path and inherit the base query."""
    if urlsplit(ref).scheme:
        return ref
    return _inherit_query(urljoin(base, ref), base)


def _normalize_entries(doc: object) -> list[dict]:
    """Reduce any accepted manifest shape to a list of entry dicts."""
    if isinstance(doc, list):
        items = doc
    elif isinstance(doc, dict):
        models = doc.get("models", doc.get("Models"))
        items = models if isinstance(models, list) else []
    else:
        items = []
    return [e for e in items if isinstance(e, dict)]


def _rebase_card(card: FinwaveModelCard, base: str) -> FinwaveModelCard:
    """Return a copy of `card` whose artifact file URLs are resolved against
    `base` (so relative URLs become absolute and inherit the base SAS)."""
    data = card.model_dump(mode="json")
    for entry in data["artifact"]["files"]:
        entry["url"] = _resolve(base, entry["url"])
    return FinwaveModelCard.model_validate(data)


def _entrypoint_sha(card: FinwaveModelCard) -> str | None:
    for f in card.artifact.files:
        if f.name == card.artifact.entrypoint:
            return f.sha256
    return None


async def _card_for_entry(
    client: httpx.AsyncClient, entry: dict, manifest_url: str
) -> FinwaveModelCard:
    """Build a rebased FinwaveModelCard from one manifest entry, fetching a
    referenced card.json when the entry is a `card_url`."""
    card_ref = entry.get("card_url") or entry.get("cardUrl")
    if card_ref:
        card_url = _resolve(manifest_url, card_ref)
        resp = await client.get(card_url)
        resp.raise_for_status()
        card = FinwaveModelCard.model_validate_json(resp.text)
        return _rebase_card(card, card_url)
    raw = entry.get("card", entry)  # inline {"card": {...}} or a bare card dict
    card = FinwaveModelCard.model_validate(raw)
    return _rebase_card(card, manifest_url)


async def warm_from_manifest(registry: ModelRegistry, settings: Settings) -> WarmupResult:
    """Fetch the configured manifest and register each model it lists. A failure
    on any one model (or the manifest fetch itself) is logged and does not abort
    the rest or the server startup; the server still serves whatever reloaded
    from the store and still accepts `/models/register`."""
    result = WarmupResult()
    url = settings.model_manifest_url
    if not url:
        return result

    try:
        async with httpx.AsyncClient(timeout=300.0, follow_redirects=True) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            entries = _normalize_entries(resp.json())
            log.info("manifest warmup: %d model(s) listed at %s", len(entries), urlsplit(url).path)
            for entry in entries:
                try:
                    card = await _card_for_entry(client, entry, url)
                except Exception as e:  # noqa: BLE001 - one bad entry must not stop the rest
                    result.failed += 1
                    log.warning("manifest warmup: could not read card from entry %s: %s", entry, e)
                    continue

                if card.model_name in registry.loaded_names() and (
                    _entrypoint_sha(card) == _entrypoint_sha(registry.loaded_card(card.model_name))
                ):
                    result.skipped += 1
                    log.info("manifest warmup: %s already loaded and current; skipping", card.model_name)
                    continue

                try:
                    await registry.register(card)
                    result.registered += 1
                    log.info("manifest warmup: registered %s (%s)", card.model_name, card.node_type)
                except Exception as e:  # noqa: BLE001
                    result.failed += 1
                    log.warning("manifest warmup: failed to register %s: %s", card.model_name, e)
    except Exception as e:  # noqa: BLE001 - manifest unreachable / malformed
        log.warning("manifest warmup: could not load manifest %s: %s", url, e)

    log.info(
        "manifest warmup complete: registered=%d skipped=%d failed=%d",
        result.registered, result.skipped, result.failed,
    )
    return result
