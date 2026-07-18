"""Audio-specific glue over the shared async-queue engine (`_queue`).

`sfx` and `music` share this: the endpoint-agnostic quote/queue/poll plumbing
lives in `_queue`; here we keep the audio content-type map and the
audio-specialized `retrieve_and_save` (optional post-processing + playback).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Callable, Optional, Tuple

from .. import audio_player
from ..client import VeniceAPIError
from . import _queue

EXT_BY_CTYPE = {
    "audio/mpeg": ".mp3",
    "audio/mp3": ".mp3",
    "audio/wav": ".wav",
    "audio/wave": ".wav",
    "audio/x-wav": ".wav",
    "audio/flac": ".flac",
}


def ext_for(ctype: str) -> Tuple[str, bool]:
    return _queue.ext_for(ctype, EXT_BY_CTYPE)


def retrieve_bytes(
    client,
    model: str,
    queue_id: str,
    *,
    poll_interval: float,
    max_wait: float,
    on_tick: Optional[Callable[[dict], None]] = None,
) -> Tuple[str, bytes]:
    """Poll /audio/retrieve until the media is ready and return (ctype, bytes).

    Print-free core of `retrieve_and_save`: the bare poll, with no file I/O,
    stdout, cleanup, or playback -- so callers that own stdout (e.g. the MCP
    stdio server) can reuse the poll without corrupting their transport. Raises
    VeniceAPIError on a terminal API error and TimeoutError on `max_wait`.
    """
    return client.poll_retrieve(
        "/audio/retrieve",
        {"model": model, "queue_id": queue_id},
        interval=poll_interval,
        max_wait=max_wait,
        on_tick=on_tick,
    )


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
    start = time.monotonic()
    try:
        ctype, audio = retrieve_bytes(
            client,
            model,
            queue_id,
            poll_interval=poll_interval,
            max_wait=max_wait,
            on_tick=_queue.progress_tick(start),
        )
    except VeniceAPIError as e:
        sys.stderr.write("\n")
        print(f"retrieve failed: {e}", file=sys.stderr)
        return _queue.status_to_exit(e)
    except TimeoutError as e:
        sys.stderr.write("\n")
        print(f"{e}; check later with: {retry_hint}", file=sys.stderr)
        return 7
    sys.stderr.write("\n")

    ext, unknown = ext_for(ctype)
    if unknown:
        print(f"warning: unexpected content-type {ctype!r}; saving as .bin", file=sys.stderr)
    out_path = _queue.resolve_output_path(out_arg, queue_id, ext, prefix=name_prefix)

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
