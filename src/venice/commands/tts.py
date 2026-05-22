"""`venice tts` -- coming soon. Stub for v0.1."""
import sys


def register(subparsers) -> None:
    p = subparsers.add_parser(
        "tts",
        help="Text-to-speech (not yet implemented).",
        description="Wraps /audio/speech. Not implemented in v0.1.",
    )
    p.set_defaults(handler=_run)


def _run(args) -> int:
    print("venice tts: not yet implemented -- coming soon.", file=sys.stderr)
    return 2
