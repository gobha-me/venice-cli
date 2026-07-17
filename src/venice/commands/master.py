"""`venice master` -- master an existing audio file with ffmpeg.

Pure local post-processing (no Venice API call): 48k/24-bit WAV, 2-pass loudnorm
(LUFS + true-peak), and an optional seamless loop. Takes files `venice music` /
`venice sfx` already produced and makes them delivery-ready.
"""
from __future__ import annotations

import sys
from pathlib import Path

from .. import audio_post


def register(subparsers) -> None:
    p = subparsers.add_parser(
        "master",
        help="Master an audio file: 48k/24-bit WAV, LUFS/true-peak, optional loop (needs ffmpeg).",
        description=(
            "Masters a local audio file with ffmpeg -- no Venice API call. Produces a "
            "WAV master (default 48kHz/24-bit) with 2-pass loudnorm (LUFS target + "
            "true-peak ceiling) and, with --loop, a seamless crossfade loop for "
            "ambience/music. Requires ffmpeg (and ffprobe for --loop) on PATH."
        ),
    )
    p.add_argument("input", type=Path, help="Input audio file (mp3/wav/flac/...).")
    p.add_argument("--output", "-o", type=Path, default=None,
                   help="Output WAV path (default: <input>.mastered.wav).")
    audio_post.add_master_flags(p, include_toggle=False)
    p.add_argument("--dry-run", action="store_true",
                   help="Print the ffmpeg commands without running them.")
    p.set_defaults(handler=_run)


def _run(args) -> int:
    if not args.input.exists():
        print(f"master: input not found: {args.input}", file=sys.stderr)
        return 6
    out = args.output or audio_post.default_output(args.input)
    return audio_post.master(
        args.input, out, dry_run=args.dry_run, **audio_post.master_kwargs(args)
    )
