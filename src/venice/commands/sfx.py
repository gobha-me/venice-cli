"""`venice sfx` -- generate a sound effect via Venice's async audio queue."""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional, Tuple

from .. import audio_player, auth, billing, config
from ..client import VeniceAPIError, build_client_from_auth

SFX_MODELS = {
    "elevenlabs-sound-effects-v2": (22, "mp3"),
    "mmaudio-v2-text-to-audio": (30, "mp3"),
}
DEFAULT_SFX_MODEL = "elevenlabs-sound-effects-v2"
DEFAULT_DURATION = 5

EXT_BY_CTYPE = {
    "audio/mpeg": ".mp3",
    "audio/mp3": ".mp3",
    "audio/wav": ".wav",
    "audio/wave": ".wav",
    "audio/x-wav": ".wav",
    "audio/flac": ".flac",
}


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


def _build_client():
    try:
        return build_client_from_auth(), 0
    except auth.AuthError as e:
        print(str(e), file=sys.stderr)
        return None, 2


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


def _ext_for(ctype: str) -> Tuple[str, bool]:
    base = (ctype or "").split(";", 1)[0].strip().lower()
    if base in EXT_BY_CTYPE:
        return EXT_BY_CTYPE[base], False
    return ".bin", True


def _resolve_output_path(arg_output: Optional[Path], queue_id: str, ext: str) -> Path:
    short = (queue_id or "unknown")[:8]
    default_name = f"venice-sfx-{short}{ext}"
    if arg_output is None:
        return Path.cwd() / default_name
    if arg_output.is_dir():
        return arg_output / default_name
    return arg_output


def _status_to_exit(err: VeniceAPIError) -> int:
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


def _print_quote(quote_usd, model: str, duration: int) -> None:
    print(
        f"Estimated cost: {billing.format_usd(quote_usd)} "
        f"(model={model}, duration={duration}s)",
        file=sys.stderr,
    )


def _print_balance_and_remaining(client, quote_usd, *, show_balance: bool) -> None:
    """Print pre-charge balance (USD + DIEM credit) and estimated remaining."""
    if not show_balance:
        return
    info = None
    try:
        info = billing.fetch_balance(client)
    except VeniceAPIError:
        info = None
    if not info or info.get("total") is None:
        return
    print(
        f"Balance:        {billing.format_balance_breakdown(info)}",
        file=sys.stderr,
    )
    try:
        remaining = float(info["total"]) - float(quote_usd)
        print(
            f"After charge:   {billing.format_usd(remaining)}",
            file=sys.stderr,
        )
    except (TypeError, ValueError):
        pass


def _confirm_or_exit(yes: bool) -> Optional[int]:
    if yes:
        return None
    if not sys.stdin.isatty():
        print(
            "non-interactive; pass --yes to confirm the charge.",
            file=sys.stderr,
        )
        return 1
    try:
        ans = input("Proceed? [y/N] ").strip().lower()
    except EOFError:
        ans = ""
    if ans not in ("y", "yes"):
        print("aborted by user", file=sys.stderr)
        return 1
    return None


def _progress_tick(start: float):
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


def _retrieve_and_save(
    client: VeniceClient,
    model: str,
    queue_id: str,
    out_arg: Optional[Path],
    poll_interval: float,
    max_wait: float,
    no_cleanup: bool,
    want_play: Optional[bool],
) -> int:
    body = {"model": model, "queue_id": queue_id}
    start = time.monotonic()
    try:
        ctype, audio = client.poll_retrieve(
            "/audio/retrieve",
            body,
            interval=poll_interval,
            max_wait=max_wait,
            on_tick=_progress_tick(start),
        )
    except VeniceAPIError as e:
        sys.stderr.write("\n")
        print(f"retrieve failed: {e}", file=sys.stderr)
        return _status_to_exit(e)
    except TimeoutError as e:
        sys.stderr.write("\n")
        print(
            f"{e}; check later with: venice sfx-status {queue_id}",
            file=sys.stderr,
        )
        return 7
    sys.stderr.write("\n")

    ext, unknown = _ext_for(ctype)
    if unknown:
        print(f"warning: unexpected content-type {ctype!r}; saving as .bin", file=sys.stderr)
    out_path = _resolve_output_path(out_arg, queue_id, ext)

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

    should_play = want_play
    if should_play is None:
        should_play = sys.stdout.isatty() and audio_player.has_player()
    if should_play:
        audio_player.play(out_path)
    return 0


def _run_generate(args) -> int:
    if not args.prompt:
        print("sfx: prompt required (or use: venice sfx-status <id>)", file=sys.stderr)
        return 2

    client, rc = _build_client()
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
        return _status_to_exit(e)

    quote_value = quote.get("quote", quote)
    _print_quote(quote_value, args.model, duration)
    _print_balance_and_remaining(
        client, quote_value, show_balance=not args.no_balance
    )

    if args.max_spend is not None:
        try:
            if float(quote_value) > float(args.max_spend):
                print(
                    f"sfx: quote {billing.format_usd(quote_value)} exceeds "
                    f"--max-spend {billing.format_usd(args.max_spend)}; aborting",
                    file=sys.stderr,
                )
                return 1
        except (TypeError, ValueError):
            pass

    if args.dry_run:
        return 0

    if not args.background:
        rc = _confirm_or_exit(args.yes)
        if rc is not None:
            return rc

    try:
        queued = client.post_json(
            "/audio/queue",
            {"model": args.model, "prompt": args.prompt, "duration_seconds": duration},
        )
    except VeniceAPIError as e:
        print(f"queue failed: {e}", file=sys.stderr)
        return _status_to_exit(e)

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

    return _retrieve_and_save(
        client,
        args.model,
        queue_id,
        args.output,
        args.poll_interval,
        args.max_wait,
        args.no_cleanup,
        args.play,
    )


def _run_status(args) -> int:
    client, rc = _build_client()
    if rc != 0:
        return rc
    want_play = True if args.play else None
    return _retrieve_and_save(
        client,
        args.model,
        args.queue_id,
        args.output,
        args.poll_interval,
        args.max_wait,
        args.no_cleanup,
        want_play,
    )
