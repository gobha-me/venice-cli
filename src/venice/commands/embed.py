"""`venice embed` -- coming soon. Stub for v0.1."""
import sys


def register(subparsers) -> None:
    p = subparsers.add_parser(
        "embed",
        help="Embeddings (not yet implemented).",
        description="Wraps /embeddings/generate. Not implemented in v0.1.",
    )
    p.set_defaults(handler=_run)


def _run(args) -> int:
    print("venice embed: not yet implemented -- coming soon.", file=sys.stderr)
    return 2
