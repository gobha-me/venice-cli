"""Shared async-queue engine for Venice's quote -> queue -> poll -> retrieve
-> complete endpoints.

Endpoint-agnostic primitives factored out of `_audio` so audio (`sfx`/`music`)
and video can share one copy of the plumbing. Callers pass primitive args
(model, queue_id, paths, an extension map) so these helpers stay independent of
any one command's argument shape.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional, Tuple

from .. import auth
from ..client import VeniceAPIError, build_client_from_auth


def build_client():
    try:
        return build_client_from_auth(), 0
    except auth.AuthError as e:
        print(str(e), file=sys.stderr)
        return None, 2


def ext_for(ctype: str, ext_map: dict, default: str = ".bin") -> Tuple[str, bool]:
    """Map a response content-type to a file extension.

    Returns (ext, unknown): `unknown` is True when the type wasn't in the map,
    in which case `default` is returned so the caller can warn.
    """
    base = (ctype or "").split(";", 1)[0].strip().lower()
    if base in ext_map:
        return ext_map[base], False
    return default, True


def resolve_output_path(
    arg_output: Optional[Path], queue_id: str, ext: str, *, prefix: str
) -> Path:
    short = (queue_id or "unknown")[:8]
    default_name = f"{prefix}-{short}{ext}"
    if arg_output is None:
        return Path.cwd() / default_name
    if arg_output.is_dir():
        return arg_output / default_name
    return arg_output


def status_to_exit(err: VeniceAPIError) -> int:
    s = err.status
    if s == 422:
        return 3
    if s == 429:
        return 4
    if 500 <= s < 600:
        return 5
    if s == 404:
        return 6
    if s == 0:
        return 8
    return 2


def progress_tick(start: float):
    def _on(payload: dict) -> None:
        avg_ms = payload.get("average_execution_time") or 0
        elapsed_ms = payload.get("execution_duration") or 0
        try:
            avg_s = float(avg_ms) / 1000.0
        except (TypeError, ValueError):
            avg_s = 0.0
        try:
            el_s = float(elapsed_ms) / 1000.0
        except (TypeError, ValueError):
            el_s = 0.0
        wall = time.monotonic() - start
        if avg_s > 0:
            sys.stderr.write(
                f"\r[wall {wall:5.1f}s | server {el_s:5.1f}s / ~{avg_s:5.1f}s] processing..."
            )
        else:
            sys.stderr.write(f"\r[wall {wall:5.1f}s] processing...")
        sys.stderr.flush()
    return _on
