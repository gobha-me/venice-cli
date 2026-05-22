"""`venice image` -- coming soon. Stub for v0.1."""
import sys


def register(subparsers) -> None:
    p = subparsers.add_parser(
        "image",
        help="Image generation (not yet implemented).",
        description="Wraps /image/generate. Not implemented in v0.1.",
    )
    p.set_defaults(handler=_run)


def _run(args) -> int:
    print("venice image: not yet implemented -- coming soon.", file=sys.stderr)
    return 2
