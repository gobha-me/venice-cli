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
from typing import Optional, Tuple

from .. import audio_player, auth, billing, config
from ..client import VeniceAPIError, build_client_from_auth

# Slugs verified against /models?type=tts on 2026-05-22.
TTS_MODELS = (
    "tts-kokoro",
    "tts-qwen3-0-6b",
    "tts-qwen3-1-7b",
    "tts-xai-v1",
    "tts-inworld-1-5-max",
    "tts-chatterbox-hd",
    "tts-orpheus",
    "tts-elevenlabs-turbo-v2-5",
    "tts-minimax-speech-02-hd",
    "tts-gemini-3-1-flash",
)
DEFAULT_TTS_MODEL = "tts-kokoro"  # cheapest ($3.50/1M chars), 54 voices

FORMATS = ("mp3", "opus", "aac", "flac", "wav", "pcm")
DEFAULT_FORMAT = "mp3"

EXT_BY_FORMAT = {
    "mp3": ".mp3",
    "opus": ".opus",
    "aac": ".aac",
    "flac": ".flac",
    "wav": ".wav",
    "pcm": ".pcm",
}


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
        choices=TTS_MODELS,
        default=DEFAULT_TTS_MODEL,
    )
    p.add_argument(
        "--voice",
        default=None,
        help="Voice id (model-specific). If omitted, Venice uses the model default.",
    )
    p.add_argument(
        "--format",
        choices=FORMATS,
        default=DEFAULT_FORMAT,
        help=f"Output audio format (default {DEFAULT_FORMAT}).",
    )
    p.add_argument(
        "--speed",
        type=float,
        default=None,
        metavar="N",
        help="Playback speed (0.25-4.0). Omit to use server default (1.0).",
    )
    p.add_argument("--output", "-o", type=Path, default=None)
    play_grp = p.add_mutually_exclusive_group()
    play_grp.add_argument("--play", dest="play", action="store_true", default=None)
    play_grp.add_argument("--no-play", dest="play", action="store_false")
    p.add_argument("--yes", "-y", action="store_true")
    p.add_argument("--dry-run", action="store_true",
                   help="Estimate cost and exit; don't call /audio/speech.")
    p.add_argument(
        "--max-spend",
        type=float,
        default=None,
        metavar="USD",
        help="Refuse to synthesize if the estimated cost exceeds this USD cap.",
    )
    p.add_argument(
        "--no-balance",
        action="store_true",
        help="Skip the upfront balance display.",
    )
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


# ---- pricing + cost estimation ----------------------------------------------

def _fetch_tts_price_per_million(client, model: str) -> Optional[float]:
    """Look up `model_spec.pricing.input.usd` for the given TTS model. Best-effort."""
    try:
        doc = client.get_json("/models", params={"type": "tts"})
    except VeniceAPIError:
        return None
    data = doc.get("data") if isinstance(doc, dict) else None
    if not isinstance(data, list):
        return None
    for m in data:
        if isinstance(m, dict) and m.get("id") == model:
            try:
                return float(m["model_spec"]["pricing"]["input"]["usd"])
            except (KeyError, TypeError, ValueError):
                return None
    return None


def _estimate_cost(char_count: int, price_per_million: Optional[float]) -> Optional[float]:
    if price_per_million is None:
        return None
    return (char_count / 1_000_000.0) * price_per_million


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
    if not info or info.get("usd") is None:
        return
    bal = info["usd"]
    print(f"Balance:        {billing.format_usd(bal)}", file=sys.stderr)
    if cost is not None:
        try:
            remaining = float(bal) - float(cost)
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

    price = _fetch_tts_price_per_million(client, args.model)
    cost = _estimate_cost(len(text), price)
    _print_estimate(cost, len(text), args.model)
    _print_balance_and_remaining(client, cost, show=not args.no_balance)

    if args.max_spend is not None and cost is not None:
        try:
            if float(cost) > float(args.max_spend):
                print(
                    f"tts: estimate {billing.format_usd(cost)} exceeds "
                    f"--max-spend {billing.format_usd(args.max_spend)}; aborting",
                    file=sys.stderr,
                )
                return 1
        except (TypeError, ValueError):
            pass

    if args.dry_run:
        return 0

    rc = _confirm_or_exit(args.yes)
    if rc is not None:
        return rc

    body: dict = {
        "input": text,
        "model": args.model,
        "response_format": args.format,
    }
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
    out_path = _resolve_output_path(args.output, short, args.format)

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
