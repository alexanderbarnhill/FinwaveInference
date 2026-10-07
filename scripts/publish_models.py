#!/usr/bin/env python3
"""Publish the local model store to a private Azure blob container as a manifest
the inference server can warm up from (FINWAVE_MODEL_MANIFEST_URL).

This is the producer side of the manifest-warmup contract (see the server's
`warmup.py`): it uploads each model's `card.json` + artifact files from the local
`_store` into `models/<model_name>/...`, rewriting the card's artifact URLs to be
*relative* (just the file name) so they resolve against the card's blob URL and
inherit whatever read SAS the server is handed, and writes a top-level
`manifest.json` that lists every model by `card_url`.

Layout produced:

    <container>/manifest.json              {"models": [{"card_url": "FIN_DETECT/card.json"}, ...]}
    <container>/FIN_DETECT/card.json       (artifact URLs rewritten to "best.onnx" etc.)
    <container>/FIN_DETECT/best.onnx
    <container>/WAKW_miew_id/card.json
    <container>/WAKW_miew_id/...

The container is PRIVATE: the Hub mints a short-lived read+list user-delegation
SAS for it at worker enroll time, so models are never public. Each artifact's
sha256 in the card is recomputed from the bytes actually uploaded, so the server's
download-time verification always matches.

Auth uses DefaultAzureCredential (so an `az login` / managed identity / service
principal all work). The identity needs `Storage Blob Data Contributor` on the
target container (and the container must exist, or the identity must be allowed to
create it).

Examples
--------
Publish every model in the default store to finwavestore/models:
    uv run --with azure-storage-blob --with azure-identity python scripts/publish_models.py \
      --account finwavestore --container models

Dry-run (list what would upload, touch nothing):
    uv run --with azure-storage-blob --with azure-identity python scripts/publish_models.py \
      --account finwavestore --container models --dry-run

Publish only two models:
    uv run ... python scripts/publish_models.py --only FIN_DETECT,WAKW_miew_id
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from urllib.parse import urlsplit

CARD_FILENAME = "card.json"
MANIFEST_SPEC_VERSION = "1.0"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _resolve_local(url: str, model_dir: Path) -> Path:
    """Resolve a card artifact URL to a local file. Cards persisted by the server
    rewrite URLs to `file://` absolute paths; a relative/bare name resolves inside
    the model dir."""
    if url.startswith("file://"):
        return Path(url[len("file://"):])
    parts = urlsplit(url)
    if parts.scheme in ("http", "https"):
        raise ValueError(f"card artifact URL is remote, not a local file: {url}")
    return model_dir / url


def _content_type(name: str) -> str:
    if name.endswith(".json"):
        return "application/json"
    return "application/octet-stream"


def _blob_size(container_client, blob_name: str) -> int | None:
    """Size of an existing blob, or None if it doesn't exist (used to skip
    already-uploaded artifacts on a resume)."""
    try:
        return container_client.get_blob_client(blob_name).get_blob_properties().size
    except Exception:
        return None


def _iter_model_dirs(store: Path, only: set[str] | None):
    for card_path in sorted(store.glob(f"*/{CARD_FILENAME}")):
        model_dir = card_path.parent
        if only is not None and model_dir.name not in only:
            continue
        yield model_dir, card_path


def publish(args: argparse.Namespace) -> int:
    store = Path(args.store)
    if not store.is_dir():
        print(f"error: model store {store} is not a directory", file=sys.stderr)
        return 2
    only = set(s.strip() for s in args.only.split(",") if s.strip()) if args.only else None

    container_client = None
    if not args.dry_run:
        from azure.identity import DefaultAzureCredential
        from azure.storage.blob import BlobServiceClient, ContentSettings  # noqa: F401

        service = BlobServiceClient(
            f"https://{args.account}.blob.core.windows.net",
            credential=DefaultAzureCredential(),
            # Large artifacts over a flaky uplink: upload in blocks (not one PUT),
            # in parallel, with retries + generous timeouts so a transient write
            # timeout doesn't abort the whole publish.
            retry_total=5, retry_connect=5, retry_read=5,
            connection_timeout=60, read_timeout=600,
            max_single_put_size=8 * 1024 * 1024,
            max_block_size=8 * 1024 * 1024,
        )
        container_client = service.get_container_client(args.container)
        try:
            container_client.create_container()
            print(f"created container {args.container}")
        except Exception:
            pass  # already exists (or no create permission; uploads will surface a real error)

    manifest_models: list[dict] = []
    published = 0
    for model_dir, card_path in _iter_model_dirs(store, only):
        name = model_dir.name
        try:
            card = json.loads(card_path.read_text())
        except Exception as e:  # noqa: BLE001
            print(f"skip {name}: unreadable card.json ({e})", file=sys.stderr)
            continue

        files = card.get("artifact", {}).get("files", [])
        local_files: list[tuple[str, Path]] = []
        ok = True
        for entry in files:
            local = _resolve_local(entry["url"], model_dir)
            if not local.is_file():
                print(f"skip {name}: missing artifact {entry['name']} ({local})", file=sys.stderr)
                ok = False
                break
            local_files.append((entry["name"], local))
            # Recompute sha256 from the bytes we will upload, and rewrite the URL
            # to a relative name so the server resolves it against the card's blob
            # URL and inherits the SAS.
            entry["sha256"] = _sha256(local)
            entry["url"] = entry["name"]
        if not ok:
            continue

        card_blob = f"{name}/{CARD_FILENAME}"
        total_bytes = sum(p.stat().st_size for _, p in local_files)
        print(f"{'DRY ' if args.dry_run else ''}publish {name}: {len(local_files)} file(s), "
              f"{total_bytes / 1e6:.1f} MB -> {args.container}/{name}/")

        if not args.dry_run:
            from azure.storage.blob import ContentSettings
            for fname, local in local_files:
                blob_name = f"{name}/{fname}"
                size = local.stat().st_size
                # Resume: skip a blob already uploaded at the same size, so a re-run
                # after a mid-publish failure only moves what's left.
                if _blob_size(container_client, blob_name) == size:
                    print(f"  skip {fname} (already uploaded, {size / 1e6:.1f} MB)")
                    continue
                with local.open("rb") as data:
                    container_client.upload_blob(
                        name=blob_name, data=data, overwrite=True, max_concurrency=4,
                        content_settings=ContentSettings(content_type=_content_type(fname)),
                    )
                print(f"  uploaded {fname} ({size / 1e6:.1f} MB)")
            container_client.upload_blob(
                name=card_blob, data=json.dumps(card, indent=2).encode(), overwrite=True,
                content_settings=ContentSettings(content_type="application/json"),
            )

        manifest_models.append({
            "key": name,
            "node_type": card.get("node_type"),
            "population_id": card.get("population_id"),
            "card_url": card_blob,
        })
        published += 1

    manifest = {
        "spec_version": MANIFEST_SPEC_VERSION,
        "generated_utc": _utc_now(),
        "models": manifest_models,
    }
    print(f"\n{'DRY ' if args.dry_run else ''}manifest: {published} model(s) -> {args.container}/{args.manifest_blob}")
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
    else:
        from azure.storage.blob import ContentSettings
        container_client.upload_blob(
            name=args.manifest_blob, data=json.dumps(manifest, indent=2).encode(), overwrite=True,
            content_settings=ContentSettings(content_type="application/json"),
        )
        print("done.")
    return 0


def _utc_now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def main() -> int:
    p = argparse.ArgumentParser(description="Publish the local model store to a private Azure blob manifest.")
    p.add_argument("--store", default=os.environ.get("FINWAVE_MODEL_STORE_PATH", "/media/alex/Storage/finwave_models/_store"),
                   help="Local model store dir (default: $FINWAVE_MODEL_STORE_PATH or the box path).")
    p.add_argument("--account", default="finwavestore", help="Azure storage account name.")
    p.add_argument("--container", default="models", help="Private container for model bundles.")
    p.add_argument("--manifest-blob", default="manifest.json", help="Manifest blob name within the container.")
    p.add_argument("--only", default=None, help="Comma-separated model names to publish (default: all in the store).")
    p.add_argument("--dry-run", action="store_true", help="List what would upload and print the manifest; touch nothing.")
    return publish(p.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
