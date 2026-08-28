"""Private local registry for server-issued VPS video download URLs.

Queue ids and models are safe public handles; presigned download URLs are not.  This
0600 atomic store binds the latter to the former across a background CLI process
boundary without returning the URL to an MCP host or model transcript.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional

from .. import _egress, config

STORE_VERSION = 1
MAX_JOBS = 100
MAX_AGE_SECONDS = 7 * 24 * 60 * 60

_LOCK = threading.Lock()


class VideoJobStoreError(Exception):
    """The private video job registry could not be read or updated."""


def _path() -> Path:
    return Path(os.environ.get(config.ENV_VIDEO_JOBS_FILE) or config.VIDEO_JOBS_FILE)


def _key(queue_id: str, model: str) -> str:
    if not isinstance(queue_id, str) or not queue_id:
        raise VideoJobStoreError("queue_id is required")
    if not isinstance(model, str) or not model:
        raise VideoJobStoreError("model is required")
    return hashlib.sha256(f"{model}\0{queue_id}".encode("utf-8")).hexdigest()


def _fresh() -> dict:
    return {"version": STORE_VERSION, "jobs": {}}


def _load(path: Path, *, strict: bool) -> dict:
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return _fresh()
    except OSError as e:
        raise VideoJobStoreError(f"cannot read private video job registry: {e}") from None
    try:
        doc = json.loads(raw)
    except ValueError as e:
        if strict:
            raise VideoJobStoreError(f"private video job registry is malformed: {e}") from None
        return _fresh()
    if (
        not isinstance(doc, dict)
        or doc.get("version") != STORE_VERSION
        or not isinstance(doc.get("jobs"), dict)
    ):
        if strict:
            raise VideoJobStoreError("private video job registry is malformed")
        return _fresh()
    return doc


def _save(path: Path, doc: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=2, sort_keys=True)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except Exception:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def _prune(doc: dict, now: float) -> bool:
    jobs = doc["jobs"]
    keep = {}
    for key, value in jobs.items():
        if not isinstance(value, dict):
            continue
        try:
            created = float(value.get("created", 0))
        except (TypeError, ValueError):
            continue
        if created > 0 and now - created <= MAX_AGE_SECONDS:
            keep[key] = value
    newest = sorted(
        keep.items(), key=lambda item: float(item[1].get("created", 0)), reverse=True
    )[:MAX_JOBS]
    pruned = dict(newest)
    changed = pruned != jobs
    doc["jobs"] = pruned
    return changed


def remember(queue_id: str, model: str, download_url: str) -> None:
    """Persist one server-issued URL without ever including it in an error."""
    if not isinstance(download_url, str) or not download_url:
        raise VideoJobStoreError("download URL is required")
    try:
        _egress.validate_https_url(download_url)
    except _egress.EgressPolicyError as e:
        raise VideoJobStoreError(str(e)) from None
    path = _path()
    now = time.time()
    with _LOCK:
        doc = _load(path, strict=True)
        _prune(doc, now)
        doc["jobs"][_key(queue_id, model)] = {
            "queue_id": queue_id,
            "model": model,
            "download_url": download_url,
            "created": now,
        }
        _prune(doc, now)
        try:
            _save(path, doc)
        except OSError as e:
            raise VideoJobStoreError(f"cannot write private video job registry: {e}") from None


def lookup(queue_id: str, model: str) -> Optional[str]:
    """Return the bound URL, pruning expired entries as a side effect."""
    path = _path()
    now = time.time()
    with _LOCK:
        doc = _load(path, strict=True)
        changed = _prune(doc, now)
        entry = doc["jobs"].get(_key(queue_id, model))
        if changed and path.exists():
            try:
                _save(path, doc)
            except OSError as e:
                raise VideoJobStoreError(
                    f"cannot update private video job registry: {e}"
                ) from None
    value = entry.get("download_url") if isinstance(entry, dict) else None
    return value if isinstance(value, str) and value else None


def forget(queue_id: str, model: str) -> None:
    path = _path()
    with _LOCK:
        doc = _load(path, strict=True)
        if doc["jobs"].pop(_key(queue_id, model), None) is None:
            return
        try:
            _save(path, doc)
        except OSError as e:
            raise VideoJobStoreError(f"cannot update private video job registry: {e}") from None
