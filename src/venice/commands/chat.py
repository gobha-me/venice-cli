"""`venice chat` -- coming soon. Stub for v0.1."""
import sys


def register(subparsers) -> None:
    p = subparsers.add_parser(
        "chat",
        help="Chat completions (not yet implemented).",
        description="Wraps /chat/completions. Not implemented in v0.1.",
    )
    p.set_defaults(handler=_run)


def _run(args) -> int:
    print("venice chat: not yet implemented -- coming soon.", file=sys.stderr)
    return 2
