"""`venice music` -- generate 60-90s ambience/music via Venice's async audio queue.

Shares the quote -> queue -> poll -> retrieve -> complete engine with `sfx`
(see `_audio`), but targets the `elevenlabs-music` model and its music-only
params. Before the paid quote it does a free `/models?type=music` lookup to
validate duration/prompt length and gate the optional params; if that lookup is
unavailable it degrades to letting the API be the backstop.
"""
from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path
from typing import Optional

from .. import audio_post, billing, config, userconfig
from ..client import VeniceAPIError
from . import _audio, _queue, _shared

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
    p.add_argument(
        "--model", default=None,
        help=f"Music model id (default {DEFAULT_MUSIC_MODEL}). Config-backable "
        "via defaults.music.model; an explicit --model still wins.",
    )
    p.add_argument(
        "--duration",
        type=int,
        default=None,
        help="Duration in seconds (omit to use the model default).",
    )
    p.add_argument(
        "--instrumental",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Force instrumental (no lyrics/vocals). Config-backable via "
        "defaults.music.instrumental; an explicit --lyrics clears a "
        "config-sourced value, and --instrumental/--no-instrumental still wins.",
    )
    p.add_argument("--lyrics", default=None, metavar="TXT", help="Lyrics prompt (lyric-capable models only).")
    p.add_argument("--speed", type=float, default=None, help="Playback speed multiplier.")
    p.add_argument("--output", "-o", type=Path, default=None)
    play_grp = p.add_mutually_exclusive_group()
    play_grp.add_argument("--play", dest="play", action="store_true", default=None)
    play_grp.add_argument("--no-play", dest="play", action="store_false")
    p.add_argument("--yes", "-y", action="store_true", default=None)
    p.add_argument("--background", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    _shared.add_cleanup_flag(p, section="music", endpoint="/audio/complete")
    p.add_argument(
        "--max-spend",
        type=float,
        default=None,
        metavar="USD",
        help="Refuse to queue if the quote exceeds this USD cap.",
    )
    _shared.add_balance_flag(p)
    _shared.add_poll_flags(p, section="music",
                           interval=config.MUSIC_POLL_INTERVAL_SEC,
                           max_wait=config.MUSIC_POLL_MAX_WAIT_SEC)
    audio_post.add_master_flags(p, include_toggle=True)
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
    # #57 Class C1: deliberately NOT tri-stated, unlike the generate parser's
    # --model. Here the model is job IDENTITY -- it goes in the /audio/retrieve
    # and /audio/complete bodies (_audio.py) -- so `defaults.music.model` must
    # never reach it and retarget an already-queued, already-charged job. The
    # concrete default is exactly what makes `apply_defaults` skip this dest
    # (it only fills a dest that is still None). Do not "tidy" it to None.
    # NOTE the --poll-interval/--max-wait pair registered just below DID become
    # `default=None` in #57 Class C2, because cadence IS a preference. The
    # inconsistency with this line is deliberate; do not reconcile them.
    sp.add_argument(
        "--model", default=DEFAULT_MUSIC_MODEL,
        help="Model the job was queued under (job identity, not a preference).",
    )
    sp.add_argument("--output", "-o", type=Path, default=None)
    status_play = sp.add_mutually_exclusive_group()
    status_play.add_argument("--play", dest="play", action="store_true",
                             default=None,
                             help="Play the audio after download. Config-backable "
                                  "via defaults.music.play.")
    status_play.add_argument("--no-play", dest="play", action="store_false",
                             default=None,
                             help="Never play (beats a config default).")
    _shared.add_cleanup_flag(sp, section="music", endpoint="/audio/complete")
    _shared.add_poll_flags(sp, section="music",
                           interval=config.MUSIC_POLL_INTERVAL_SEC,
                           max_wait=config.MUSIC_POLL_MAX_WAIT_SEC)
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


def resolve_instrumental(instrumental, lyrics, *, explicit: bool):
    """A CONFIG-sourced `--instrumental` yields to a deliberate `--lyrics` (#57).

    `lyrics` is deliberately excluded from `_COMMAND_MAP` (it's per-song content,
    not a preference), so it can only ever have come from the CLI or a tool call
    -- it is always deliberate. `instrumental` may have come from
    `defaults.music.instrumental`, and `apply_defaults` runs BEFORE `_validate`,
    so without this a user who set that key once and then typed only `--lyrics`
    would hit the mutual-exclusion error for a flag they never passed.

    Only when BOTH were deliberate does `_validate`'s exit 2 still fire.
    """
    if instrumental and lyrics and not explicit:
        print("music: --lyrics overrides defaults.music.instrumental for this run",
              file=sys.stderr)
        return False
    return instrumental


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
    # Capture explicitness BEFORE config can fill the dest -- afterwards a
    # config-sourced True is indistinguishable from a typed --instrumental.
    instrumental_explicit = args.instrumental is not None
    userconfig.apply_defaults(args, "music")
    args.instrumental = resolve_instrumental(
        args.instrumental, args.lyrics, explicit=instrumental_explicit
    )
    # #57 Class C1: built-in literal last, so config gets first refusal.
    # Position relative to resolve_instrumental is not load-bearing -- it reads
    # instrumental/lyrics, never model -- so this may move if that changes.
    userconfig.apply_literals(args, model=DEFAULT_MUSIC_MODEL)
    # #57 Class C2: the mastering chain's literals. Evaluated lazily by
    # `master_hook` at post-save time, but filled here so the namespace is whole.
    audio_post.apply_master_literals(args)
    _shared.apply_poll_defaults(args, label="music",
                                interval=config.MUSIC_POLL_INTERVAL_SEC,
                                max_wait=config.MUSIC_POLL_MAX_WAIT_SEC)
    if not args.prompt:
        print("music: prompt required (or use: venice music-status <id>)", file=sys.stderr)
        return 2

    if args.master and not audio_post.has_ffmpeg():
        print("music: mastering requires ffmpeg on PATH; install it, pass "
              "--no-master, or unset defaults.music.master",
              file=sys.stderr)
        return 2

    client, rc = _queue.build_client()
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
        return _queue.status_to_exit(e)

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
        return _queue.status_to_exit(e)

    queue_id = queued.get("queue_id") or queued.get("id") or ""
    if not queue_id:
        print(f"queue response missing queue_id: {queued!r}", file=sys.stderr)
        return 5

    if args.background:
        sys.stdout.write(queue_id + "\n")
        sys.stdout.flush()
        print(
            # The model may have come from defaults.music.model rather than the
            # command line, and on the status side it is job identity, so the
            # hint has to carry it. (#57 Class C1)
            f"queued as {queue_id}; fetch with: "
            f"venice music-status {queue_id} --model {shlex.quote(args.model)}",
            file=sys.stderr,
        )
        return 0

    post = audio_post.master_hook(args) if args.master else None
    return _audio.retrieve_and_save(
        client,
        args.model,
        queue_id,
        args.output,
        args.poll_interval,
        args.max_wait,
        bool(args.no_cleanup),
        args.play,
        name_prefix="venice-music",
        retry_hint=f"venice music-status {queue_id} --model {shlex.quote(args.model)}",
        post_process=post,
    )


def _run_status(args) -> int:
    # #57 Class B: status shares its parent's section, so a config
    # default like defaults.music.no_cleanup applies to `venice music`
    # and `venice music-status` alike rather than only the former.
    userconfig.apply_defaults(args, "music")
    # #57 Class C2: the cadence literals. This parser has no mastering flags
    # (status never masters), so only the poll pair needs filling here.
    _shared.apply_poll_defaults(args, label="music-status",
                                interval=config.MUSIC_POLL_INTERVAL_SEC,
                                max_wait=config.MUSIC_POLL_MAX_WAIT_SEC)
    client, rc = _queue.build_client()
    if rc != 0:
        return rc
    # None = auto-detect (tty + a player); the tri-state maps 1:1 onto
    # `retrieve_and_save`'s Optional[bool] want_play. (#57 Class B)
    want_play = args.play
    return _audio.retrieve_and_save(
        client,
        args.model,
        args.queue_id,
        args.output,
        args.poll_interval,
        args.max_wait,
        bool(args.no_cleanup),
        want_play,
        name_prefix="venice-music",
        retry_hint=f"venice music-status {args.queue_id} --model {shlex.quote(args.model)}",
    )
