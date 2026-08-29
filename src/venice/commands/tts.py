"""`venice tts` -- synthesize speech via Venice's /audio/speech endpoint.

Sync flow (no queue). Pricing is per 1M characters; the command fetches
the live per-model rate from /models?type=tts and shows the estimate
upfront alongside the current balance. Mirrors the SFX UX where
sensible (--yes, --max-spend, --no-balance, --dry-run, --output, --play).
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import NamedTuple, Optional, Tuple

from .. import _numeric, audio_player, auth, billing, config, userconfig
from ..client import VeniceAPIError, build_client_from_auth
from . import _shared

DEFAULT_TTS_MODEL = "tts-kokoro"

EXT_BY_FORMAT = {
    "mp3": ".mp3",
    "opus": ".opus",
    "aac": ".aac",
    "flac": ".flac",
    "wav": ".wav",
    "pcm": ".pcm",
}

FORMAT_BY_CTYPE = {
    "audio/mpeg": "mp3",
    "audio/mp3": "mp3",
    "audio/opus": "opus",
    "audio/ogg": "opus",
    "audio/aac": "aac",
    "audio/flac": "flac",
    "audio/wav": "wav",
    "audio/wave": "wav",
    "audio/x-wav": "wav",
    "audio/pcm": "pcm",
    "audio/l16": "pcm",
}


class ResolvedTTS(NamedTuple):
    price_per_million: Optional[float]
    supported_formats: Tuple[str, ...]
    requested_format: Optional[str]
    output_format: str


def register(subparsers) -> None:
    p = subparsers.add_parser(
        "tts",
        help="Synthesize speech via /audio/speech (sync).",
        description=(
            "Synthesizes speech. Input from positional text, --from-file, "
            "or --stdin. Pricing is per 1M characters; cost is estimated "
            "from the live model rate. Use `venice models <model-slug>` to "
            "see the voice list for a given TTS model."
        ),
    )
    p.add_argument(
        "text",
        nargs="?",
        help="Text to speak. Use '-' for stdin, or omit and pass --from-file/--stdin.",
    )
    src = p.add_mutually_exclusive_group()
    src.add_argument(
        "--from-file",
        type=Path,
        default=None,
        metavar="PATH",
        help="Read the input text from PATH instead of the positional arg.",
    )
    src.add_argument(
        "--stdin",
        action="store_true",
        help="Read the input text from stdin until EOF.",
    )
    p.add_argument(
        "--model",
        default=None,
        help=f"TTS model id, validated against the live catalog (default "
        f"{DEFAULT_TTS_MODEL}). Config-backable via defaults.tts.model; an "
        "explicit --model still wins.",
    )
    p.add_argument(
        "--voice",
        default=None,
        help="Voice id (model-specific). If omitted, Venice uses the model default.",
    )
    p.add_argument(
        "--format",
        default=None,
        help="Model-specific output audio format. If omitted, Venice uses the "
        "selected model's live catalog default. Config-backable via "
        "defaults.tts.format; an explicit --format still wins.",
    )
    p.add_argument(
        "--speed",
        type=_numeric.finite_float,
        default=None,
        metavar="N",
        help="Playback speed (0.25-4.0). Omit to use server default (1.0).",
    )
    p.add_argument("--output", "-o", type=Path, default=None)
    play_grp = p.add_mutually_exclusive_group()
    play_grp.add_argument("--play", dest="play", action="store_true", default=None)
    play_grp.add_argument("--no-play", dest="play", action="store_false")
    p.add_argument("--yes", "-y", action="store_true", default=None)
    p.add_argument("--dry-run", action="store_true",
                   help="Estimate cost and exit; don't call /audio/speech.")
    p.add_argument(
        "--max-spend",
        type=_numeric.non_negative_float,
        default=None,
        metavar="USD",
        help="Refuse to synthesize if the estimated cost exceeds this USD cap.",
    )
    _shared.add_balance_flag(p)
    p.set_defaults(handler=_run)


# ---- input source resolution -------------------------------------------------

def _read_input(args) -> Tuple[Optional[str], int]:
    """Resolve the input text. Returns (text, exit_code). text=None on error."""
    sources = sum(
        1 for v in (args.text and args.text != "-", args.from_file, args.stdin) if v
    )
    use_stdin = bool(args.stdin) or args.text == "-"

    if args.from_file:
        if args.text is not None and args.text != "-":
            print("tts: cannot combine positional text with --from-file", file=sys.stderr)
            return None, 2
        try:
            text = args.from_file.read_text(encoding="utf-8")
        except OSError as e:
            print(f"tts: cannot read {args.from_file}: {e}", file=sys.stderr)
            return None, 2
    elif use_stdin:
        text = sys.stdin.read()
    elif args.text:
        text = args.text
    else:
        print(
            "tts: input required (positional text, --from-file PATH, or --stdin)",
            file=sys.stderr,
        )
        return None, 2

    text = text.strip()
    if not text:
        print("tts: input is empty", file=sys.stderr)
        return None, 2
    return text, 0


# ---- live model resolution + cost estimation --------------------------------

def _resolve_tts(client, model: str, requested_format: Optional[str]) -> ResolvedTTS:
    """Resolve one TTS model and its format before confirmation or spend."""
    try:
        doc = client.get_json("/models", params={"type": "tts"})
    except VeniceAPIError as e:
        raise ValueError(f"could not fetch live TTS catalog: {e}") from None
    data = doc.get("data") if isinstance(doc, dict) else None
    if not isinstance(data, list):
        raise ValueError("live TTS catalog has no data list")
    found = None
    for m in data:
        if isinstance(m, dict) and m.get("id") == model:
            found = m
            break
    if found is None:
        raise ValueError(f"model {model!r} is not in the live TTS catalog")

    spec = found.get("model_spec")
    if not isinstance(spec, dict):
        raise ValueError(f"live TTS catalog entry for {model!r} has no model_spec")
    raw_formats = spec.get("supported_formats")
    if (
        not isinstance(raw_formats, list)
        or not raw_formats
        or any(not isinstance(v, str) or not v.strip() for v in raw_formats)
    ):
        raise ValueError(
            f"live TTS catalog entry for {model!r} has invalid supported_formats"
        )
    supported = tuple(v.strip() for v in raw_formats)

    chosen = requested_format.strip() if isinstance(requested_format, str) else None
    if requested_format is not None and not chosen:
        raise ValueError("format must not be empty")
    if chosen is not None:
        if chosen not in supported:
            raise ValueError(
                f"format {chosen!r} is not supported by {model!r}; choose from "
                f"{', '.join(supported)}"
            )
        output_format = chosen
    else:
        default_format = spec.get("default_format")
        if not isinstance(default_format, str) or not default_format.strip():
            raise ValueError(
                f"live TTS catalog entry for {model!r} has no default_format"
            )
        output_format = default_format.strip()
        if output_format not in supported:
            raise ValueError(
                f"live TTS catalog default_format {output_format!r} for {model!r} "
                "is not in supported_formats"
            )

    price = None
    try:
        raw_price = spec["pricing"]["input"]["usd"]
    except (KeyError, TypeError):
        pass
    else:
        try:
            price = _numeric.non_negative_float(raw_price)
        except (TypeError, ValueError):
            raise ValueError("model catalog contains an invalid TTS price") from None
    return ResolvedTTS(price, supported, chosen, output_format)


def _estimate_cost(char_count: int, price_per_million: Optional[float]) -> Optional[float]:
    if price_per_million is None:
        return None
    return _numeric.non_negative_float(
        (char_count / 1_000_000.0) * price_per_million
    )


# ---- output path -------------------------------------------------------------

def _short_id(text: str, model: str, voice: Optional[str]) -> str:
    """Stable 8-char hex tag derived from inputs; useful as a filename suffix."""
    h = hashlib.sha1()
    h.update(text.encode("utf-8"))
    h.update(model.encode("utf-8"))
    if voice:
        h.update(voice.encode("utf-8"))
    return h.hexdigest()[:8]


def _resolve_output_path(arg_output: Optional[Path], short: str, fmt: str) -> Path:
    ext = EXT_BY_FORMAT.get(fmt, ".bin")
    default_name = f"venice-tts-{short}{ext}"
    if arg_output is None:
        return Path.cwd() / default_name
    if arg_output.is_dir():
        return arg_output / default_name
    return arg_output


def _response_format(ctype: str, fallback: str) -> str:
    base = (ctype or "").split(";", 1)[0].strip().lower()
    return FORMAT_BY_CTYPE.get(base, fallback)


# ---- main flow ---------------------------------------------------------------

def _print_estimate(cost: Optional[float], char_count: int, model: str) -> None:
    if cost is None:
        print(
            f"Estimated cost: (unknown — could not fetch {model} pricing) "
            f"[{char_count} chars]",
            file=sys.stderr,
        )
    else:
        print(
            f"Estimated cost: {billing.format_usd(cost)} "
            f"({char_count} chars, model={model})",
            file=sys.stderr,
        )


def _print_balance_and_remaining(client, cost: Optional[float], *, show: bool) -> None:
    if not show:
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
    if cost is not None:
        try:
            remaining = float(info["total"]) - float(cost)
            print(f"After charge:   {billing.format_usd(remaining)}", file=sys.stderr)
        except (TypeError, ValueError):
            pass


def _confirm_or_exit(yes: bool) -> Optional[int]:
    if yes:
        return None
    if not sys.stdin.isatty():
        print("non-interactive; pass --yes to confirm the charge.", file=sys.stderr)
        return 1
    try:
        ans = input("Proceed? [y/N] ").strip().lower()
    except EOFError:
        ans = ""
    if ans not in ("y", "yes"):
        print("aborted by user", file=sys.stderr)
        return 1
    return None


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


def _validate_speed(speed: Optional[float]) -> Optional[int]:
    if speed is None:
        return None
    if not (0.25 <= speed <= 4.0):
        print(f"tts: --speed {speed} out of range (0.25-4.0)", file=sys.stderr)
        return 2
    return None


def _run(args) -> int:
    userconfig.apply_defaults(args, "tts")
    # #57 Class C1: the built-in model literal, after config has had its turn.
    # Format deliberately stays None so the API can apply the model-specific default.
    userconfig.apply_literals(args, model=DEFAULT_TTS_MODEL)
    rc = _validate_speed(args.speed)
    if rc is not None:
        return rc

    text, rc = _read_input(args)
    if text is None:
        return rc

    try:
        client = build_client_from_auth()
    except auth.AuthError as e:
        print(str(e), file=sys.stderr)
        return 2

    try:
        resolved = _resolve_tts(client, args.model, args.format)
    except ValueError as e:
        print(f"tts: {e}", file=sys.stderr)
        return 2
    cost = _estimate_cost(len(text), resolved.price_per_million)
    _print_estimate(cost, len(text), args.model)
    _print_balance_and_remaining(client, cost, show=not args.no_balance)

    if _shared.over_budget(cost, args.max_spend):
        shown = billing.format_usd(cost) if cost is not None else "unknown"
        print(
            f"tts: estimate {shown} cannot be kept within "
            f"--max-spend {billing.format_usd(args.max_spend)}; aborting",
            file=sys.stderr,
        )
        return 1

    if args.dry_run:
        return 0

    rc = _confirm_or_exit(args.yes)
    if rc is not None:
        return rc

    body: dict = {
        "input": text,
        "model": args.model,
    }
    if resolved.requested_format is not None:
        body["response_format"] = resolved.requested_format
    if args.voice:
        body["voice"] = args.voice
    if args.speed is not None:
        body["speed"] = args.speed

    try:
        status, ctype, audio = client.request(
            "POST", "/audio/speech", json_body=body
        )
    except VeniceAPIError as e:
        print(f"tts failed: {e}", file=sys.stderr)
        return _status_to_exit(e)

    if not audio:
        print("tts: server returned empty body", file=sys.stderr)
        return 5

    short = _short_id(text, args.model, args.voice)
    output_format = _response_format(ctype, resolved.output_format)
    out_path = _resolve_output_path(args.output, short, output_format)

    try:
        out_path.write_bytes(audio)
    except OSError as e:
        print(f"could not write {out_path}: {e}", file=sys.stderr)
        return 9

    abs_path = out_path.resolve()
    print(str(abs_path))
    print(f"wrote {len(audio)} bytes to {abs_path}", file=sys.stderr)

    should_play = args.play
    if should_play is None:
        should_play = sys.stdout.isatty() and audio_player.has_player()
    if should_play:
        audio_player.play(out_path)
    return 0
