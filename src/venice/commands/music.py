"""`venice music` -- generate 60-90s ambience/music via Venice's async audio queue.

Shares the quote -> queue -> poll -> retrieve -> complete engine with `sfx`
(see `_audio`), but targets the `elevenlabs-music` model and its music-only
params. Before the paid quote it does a free `/models?type=music` lookup to
validate duration/prompt length and gate the optional params; if that lookup is
unavailable it degrades to letting the API be the backstop.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from .. import billing, config
from ..client import VeniceAPIError
from . import _audio, _shared

DEFAULT_MUSIC_MODEL = "elevenlabs-music"


def register(subparsers) -> None:
    p = subparsers.add_parser(
        "music",
        help="Generate 60-90s ambience/music via Venice audio queue.",
        description=(
            "Generates long-form music/ambience with the elevenlabs-music model. "
            "Async flow: quote -> queue -> poll -> save. Use --dry-run to see only "
            "the cost quote. To fetch a backgrounded job by its queue_id, use "
            "`venice music-status`."
        ),
    )
    p.add_argument("prompt", nargs="?", help="Music/ambience description (e.g. 'tense dungeon drone').")
    p.add_argument("--model", default=DEFAULT_MUSIC_MODEL)
    p.add_argument(
        "--duration",
        type=int,
        default=None,
        help="Duration in seconds (omit to use the model default).",
    )
    p.add_argument(
        "--instrumental",
        action="store_true",
        help="Force instrumental (no lyrics/vocals).",
    )
    p.add_argument("--lyrics", default=None, metavar="TXT", help="Lyrics prompt (lyric-capable models only).")
    p.add_argument("--speed", type=float, default=None, help="Playback speed multiplier.")
    p.add_argument("--output", "-o", type=Path, default=None)
    play_grp = p.add_mutually_exclusive_group()
    play_grp.add_argument("--play", dest="play", action="store_true", default=None)
    play_grp.add_argument("--no-play", dest="play", action="store_false")
    p.add_argument("--yes", "-y", action="store_true")
    p.add_argument("--background", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--no-cleanup", action="store_true")
    p.add_argument(
        "--max-spend",
        type=float,
        default=None,
        metavar="USD",
        help="Refuse to queue if the quote exceeds this USD cap.",
    )
    p.add_argument(
        "--no-balance",
        action="store_true",
        help="Skip the upfront balance display.",
    )
    p.add_argument("--poll-interval", type=float, default=config.SFX_POLL_INTERVAL_SEC)
    p.add_argument("--max-wait", type=float, default=config.SFX_POLL_MAX_WAIT_SEC)
    p.set_defaults(handler=_run_generate)


def register_status(subparsers) -> None:
    sp = subparsers.add_parser(
        "music-status",
        help="Fetch a previously-backgrounded music job by queue_id.",
        description=(
            "Polls /audio/retrieve for an already-queued job (typically from "
            "`venice music ... --background`) and downloads the audio when ready."
        ),
    )
    sp.add_argument("queue_id")
    sp.add_argument("--model", default=DEFAULT_MUSIC_MODEL)
    sp.add_argument("--output", "-o", type=Path, default=None)
    sp.add_argument("--play", action="store_true")
    sp.add_argument("--no-cleanup", action="store_true")
    sp.add_argument("--poll-interval", type=float, default=config.SFX_POLL_INTERVAL_SEC)
    sp.add_argument("--max-wait", type=float, default=config.SFX_POLL_MAX_WAIT_SEC)
    sp.set_defaults(handler=_run_status)


def fetch_music_spec(client, model_id: str) -> Optional[dict]:
    """Best-effort fetch of a music model's `model_spec` from /models.

    Returns the spec dict, or None if the catalog can't be fetched or the model
    isn't present (caller then skips client-side validation)."""
    try:
        doc = client.get_json("/models", params={"type": "music"})
    except VeniceAPIError:
        return None
    data = doc.get("data") if isinstance(doc, dict) else None
    if not isinstance(data, list):
        return None
    for m in data:
        if isinstance(m, dict) and m.get("id") == model_id:
            spec = m.get("model_spec")
            return spec if isinstance(spec, dict) else {}
    return None


def _norm_meta(spec: dict) -> dict:
    """Flatten model_spec + its capabilities into a case/underscore-insensitive
    lookup (the API returns camelCase capability keys; the spec uses snake_case)."""
    meta: dict = {}
    for k, v in spec.items():
        if k == "capabilities":
            continue
        meta[k.lower().replace("_", "")] = v
    caps = spec.get("capabilities")
    if isinstance(caps, dict):
        for k, v in caps.items():
            meta[k.lower().replace("_", "")] = v
    return meta


def _num(v):
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    return None


def _validate(args, spec: Optional[dict]) -> Optional[int]:
    """Return an exit code if the request is invalid, else None. Music-only
    params are gated by the model's advertised capabilities."""
    if args.lyrics and args.instrumental:
        print("music: --lyrics and --instrumental are mutually exclusive", file=sys.stderr)
        return 2

    if spec is None:
        print(
            "music: could not fetch model metadata; skipping validation "
            "(the API will reject invalid requests)",
            file=sys.stderr,
        )
        return None

    meta = _norm_meta(spec)

    minp = _num(meta.get("minpromptlength"))
    maxp = _num(meta.get("promptcharacterlimit"))
    plen = len(args.prompt or "")
    if maxp is not None and plen > maxp:
        print(f"music: prompt is {plen} chars; max is {int(maxp)}", file=sys.stderr)
        return 2
    if minp is not None and plen < minp:
        print(f"music: prompt is {plen} chars; min is {int(minp)}", file=sys.stderr)
        return 2

    if args.duration is not None:
        opts = meta.get("durationoptions")
        if isinstance(opts, list) and opts:
            if args.duration not in opts:
                allowed = ", ".join(str(o) for o in opts)
                print(f"music: --duration {args.duration}s not allowed; options: {allowed}", file=sys.stderr)
                return 2
        else:
            mind = _num(meta.get("minduration"))
            maxd = _num(meta.get("maxduration"))
            if mind is not None and args.duration < mind:
                print(f"music: --duration {args.duration}s below model min {int(mind)}s", file=sys.stderr)
                return 2
            if maxd is not None and args.duration > maxd:
                print(f"music: --duration {args.duration}s above model max {int(maxd)}s", file=sys.stderr)
                return 2

    if args.instrumental and meta.get("supportsforceinstrumental") is False:
        print(f"music: {args.model} does not support --instrumental", file=sys.stderr)
        return 2

    if args.lyrics and meta.get("supportslyrics") is False:
        print(f"music: {args.model} does not support --lyrics", file=sys.stderr)
        return 2

    if args.speed is not None:
        if meta.get("supportsspeed") is False:
            print(f"music: {args.model} does not support --speed", file=sys.stderr)
            return 2
        mins = _num(meta.get("minspeed"))
        maxs = _num(meta.get("maxspeed"))
        if mins is not None and args.speed < mins:
            print(f"music: --speed {args.speed} below model min {mins}", file=sys.stderr)
            return 2
        if maxs is not None and args.speed > maxs:
            print(f"music: --speed {args.speed} above model max {maxs}", file=sys.stderr)
            return 2

    return None


def _run_generate(args) -> int:
    if not args.prompt:
        print("music: prompt required (or use: venice music-status <id>)", file=sys.stderr)
        return 2

    client, rc = _audio.build_client()
    if rc != 0:
        return rc

    spec = fetch_music_spec(client, args.model)
    rc = _validate(args, spec)
    if rc is not None:
        return rc

    quote_body = {"model": args.model}
    if args.duration is not None:
        quote_body["duration_seconds"] = args.duration
    try:
        quote = client.post_json("/audio/quote", quote_body)
    except VeniceAPIError as e:
        print(f"quote rejected: {e}", file=sys.stderr)
        return _audio.status_to_exit(e)

    quote_value = quote.get("quote", quote)
    label = f"model={args.model}"
    if args.duration is not None:
        label += f", duration={args.duration}s"
    _shared.print_estimate(quote_value, label)
    _shared.print_balance_and_remaining(client, quote_value, show=not args.no_balance)

    if _shared.over_budget(quote_value, args.max_spend):
        print(
            f"music: quote {billing.format_usd(quote_value)} exceeds "
            f"--max-spend {billing.format_usd(args.max_spend)}; aborting",
            file=sys.stderr,
        )
        return 1

    if args.dry_run:
        return 0

    if not args.background:
        rc = _shared.confirm_or_exit(args.yes)
        if rc is not None:
            return rc

    queue_body = {"model": args.model, "prompt": args.prompt}
    if args.duration is not None:
        queue_body["duration_seconds"] = args.duration
    if args.instrumental:
        queue_body["force_instrumental"] = True
    if args.lyrics:
        queue_body["lyrics_prompt"] = args.lyrics
    if args.speed is not None:
        queue_body["speed"] = args.speed

    try:
        queued = client.post_json("/audio/queue", queue_body)
    except VeniceAPIError as e:
        print(f"queue failed: {e}", file=sys.stderr)
        return _audio.status_to_exit(e)

    queue_id = queued.get("queue_id") or queued.get("id") or ""
    if not queue_id:
        print(f"queue response missing queue_id: {queued!r}", file=sys.stderr)
        return 5

    if args.background:
        sys.stdout.write(queue_id + "\n")
        sys.stdout.flush()
        print(
            f"queued as {queue_id}; fetch with: venice music-status {queue_id}",
            file=sys.stderr,
        )
        return 0

    return _audio.retrieve_and_save(
        client,
        args.model,
        queue_id,
        args.output,
        args.poll_interval,
        args.max_wait,
        args.no_cleanup,
        args.play,
        name_prefix="venice-music",
        retry_hint=f"venice music-status {queue_id}",
    )


def _run_status(args) -> int:
    client, rc = _audio.build_client()
    if rc != 0:
        return rc
    want_play = True if args.play else None
    return _audio.retrieve_and_save(
        client,
        args.model,
        args.queue_id,
        args.output,
        args.poll_interval,
        args.max_wait,
        args.no_cleanup,
        want_play,
        name_prefix="venice-music",
        retry_hint=f"venice music-status {args.queue_id}",
    )
