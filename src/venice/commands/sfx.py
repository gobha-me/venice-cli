"""`venice sfx` -- generate a sound effect via Venice's async audio queue."""
from __future__ import annotations

import sys
from pathlib import Path

from .. import audio_post, billing, config, userconfig
from ..client import VeniceAPIError
from . import _audio, _queue, _shared

SFX_MODELS = {
    "elevenlabs-sound-effects-v2": (22, "mp3"),
    "mmaudio-v2-text-to-audio": (30, "mp3"),
}
DEFAULT_SFX_MODEL = "elevenlabs-sound-effects-v2"
DEFAULT_DURATION = 5


def register(subparsers) -> None:
    p = subparsers.add_parser(
        "sfx",
        help="Generate a sound effect via Venice audio queue.",
        description=(
            "Generates a sound effect. Async flow: quote -> queue -> poll -> "
            "save. Use --dry-run to see only the cost quote. To fetch a "
            "backgrounded job by its queue_id, use `venice sfx-status`."
        ),
    )
    p.add_argument("prompt", nargs="?", help="Sound description (e.g. 'thunderstorm rolling in').")
    p.add_argument(
        "--model",
        choices=sorted(SFX_MODELS),
        default=DEFAULT_SFX_MODEL,
    )
    p.add_argument("--duration", type=int, default=DEFAULT_DURATION)
    p.add_argument("--output", "-o", type=Path, default=None)
    play_grp = p.add_mutually_exclusive_group()
    play_grp.add_argument("--play", dest="play", action="store_true", default=None)
    play_grp.add_argument("--no-play", dest="play", action="store_false")
    p.add_argument("--yes", "-y", action="store_true", default=None)
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
    audio_post.add_master_flags(p, include_toggle=True)
    p.set_defaults(handler=_run_generate)


def register_status(subparsers) -> None:
    sp = subparsers.add_parser(
        "sfx-status",
        help="Fetch a previously-backgrounded SFX job by queue_id.",
        description=(
            "Polls /audio/retrieve for an already-queued job (typically from "
            "`venice sfx ... --background`) and downloads the audio when ready."
        ),
    )
    sp.add_argument("queue_id")
    sp.add_argument(
        "--model",
        choices=sorted(SFX_MODELS),
        default=DEFAULT_SFX_MODEL,
    )
    sp.add_argument("--output", "-o", type=Path, default=None)
    sp.add_argument("--play", action="store_true")
    sp.add_argument("--no-cleanup", action="store_true")
    sp.add_argument("--poll-interval", type=float, default=config.SFX_POLL_INTERVAL_SEC)
    sp.add_argument("--max-wait", type=float, default=config.SFX_POLL_MAX_WAIT_SEC)
    sp.set_defaults(handler=_run_status)


def _clamp_duration(model: str, duration: int) -> int:
    max_dur, _ = SFX_MODELS[model]
    if duration <= 0:
        print(f"sfx: --duration must be > 0; using {DEFAULT_DURATION}", file=sys.stderr)
        return DEFAULT_DURATION
    if duration > max_dur:
        print(
            f"sfx: --duration {duration}s exceeds {model} max {max_dur}s; clamping",
            file=sys.stderr,
        )
        return max_dur
    return duration


def _run_generate(args) -> int:
    userconfig.apply_defaults(args, "sfx")
    if not args.prompt:
        print("sfx: prompt required (or use: venice sfx-status <id>)", file=sys.stderr)
        return 2

    if args.master and not audio_post.has_ffmpeg():
        print("sfx: --master requires ffmpeg on PATH; install it or drop --master",
              file=sys.stderr)
        return 2

    client, rc = _queue.build_client()
    if rc != 0:
        return rc

    duration = _clamp_duration(args.model, args.duration)

    try:
        quote = client.post_json(
            "/audio/quote",
            {"model": args.model, "duration_seconds": duration},
        )
    except VeniceAPIError as e:
        print(f"quote rejected: {e}", file=sys.stderr)
        return _queue.status_to_exit(e)

    quote_value = quote.get("quote", quote)
    _shared.print_estimate(quote_value, f"model={args.model}, duration={duration}s")
    _shared.print_balance_and_remaining(client, quote_value, show=not args.no_balance)

    if _shared.over_budget(quote_value, args.max_spend):
        print(
            f"sfx: quote {billing.format_usd(quote_value)} exceeds "
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

    try:
        queued = client.post_json(
            "/audio/queue",
            {"model": args.model, "prompt": args.prompt, "duration_seconds": duration},
        )
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
            f"queued as {queue_id}; fetch with: venice sfx-status {queue_id}",
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
        args.no_cleanup,
        args.play,
        name_prefix="venice-sfx",
        retry_hint=f"venice sfx-status {queue_id}",
        post_process=post,
    )


def _run_status(args) -> int:
    client, rc = _queue.build_client()
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
        name_prefix="venice-sfx",
        retry_hint=f"venice sfx-status {args.queue_id}",
    )
