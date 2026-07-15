"""Shared async-audio engine for queue-based audio commands.

Extracted from `sfx` so it and `music` share one copy of the
quote -> queue -> poll -> retrieve -> complete plumbing. The functions take
primitive args (model, queue_id, paths) plus a `name_prefix`/`retry_hint` so
they stay independent of any one command's argument shape.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Callable, Optional, Tuple

from .. import audio_player, auth
from ..client import VeniceAPIError, build_client_from_auth

EXT_BY_CTYPE = {
    "audio/mpeg": ".mp3",
    "audio/mp3": ".mp3",
    "audio/wav": ".wav",
    "audio/wave": ".wav",
    "audio/x-wav": ".wav",
    "audio/flac": ".flac",
}


def build_client():
    try:
        return build_client_from_auth(), 0
    except auth.AuthError as e:
        print(str(e), file=sys.stderr)
        return None, 2


def ext_for(ctype: str) -> Tuple[str, bool]:
    base = (ctype or "").split(";", 1)[0].strip().lower()
    if base in EXT_BY_CTYPE:
        return EXT_BY_CTYPE[base], False
    return ".bin", True


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


def retrieve_and_save(
    client,
    model: str,
    queue_id: str,
    out_arg: Optional[Path],
    poll_interval: float,
    max_wait: float,
    no_cleanup: bool,
    want_play: Optional[bool],
    *,
    name_prefix: str,
    retry_hint: str,
    post_process: Optional[Callable[[Path], int]] = None,
) -> int:
    body = {"model": model, "queue_id": queue_id}
    start = time.monotonic()
    try:
        ctype, audio = client.poll_retrieve(
            "/audio/retrieve",
            body,
            interval=poll_interval,
            max_wait=max_wait,
            on_tick=progress_tick(start),
        )
    except VeniceAPIError as e:
        sys.stderr.write("\n")
        print(f"retrieve failed: {e}", file=sys.stderr)
        return status_to_exit(e)
    except TimeoutError as e:
        sys.stderr.write("\n")
        print(f"{e}; check later with: {retry_hint}", file=sys.stderr)
        return 7
    sys.stderr.write("\n")

    ext, unknown = ext_for(ctype)
    if unknown:
        print(f"warning: unexpected content-type {ctype!r}; saving as .bin", file=sys.stderr)
    out_path = resolve_output_path(out_arg, queue_id, ext, prefix=name_prefix)

    try:
        out_path.write_bytes(audio)
    except OSError as e:
        print(f"could not write {out_path}: {e}", file=sys.stderr)
        return 9

    abs_path = out_path.resolve()
    print(str(abs_path))
    print(f"wrote {len(audio)} bytes to {abs_path}", file=sys.stderr)

    if not no_cleanup:
        try:
            client.post_json("/audio/complete", {"model": model, "queue_id": queue_id})
        except VeniceAPIError as e:
            print(f"warning: cleanup call failed: {e}", file=sys.stderr)

    post_rc = 0
    if post_process is not None:
        post_rc = post_process(out_path)

    should_play = want_play
    if should_play is None:
        should_play = sys.stdout.isatty() and audio_player.has_player()
    if should_play:
        audio_player.play(out_path)
    return post_rc
