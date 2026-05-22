"""Top-level argparse dispatcher."""
import argparse
import sys

from . import __version__
from .commands import register_all


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="venice",
        description="Venice.ai CLI (stdlib-only). `venice <command> --help` for details.",
    )
    p.add_argument("--version", action="version", version=f"venice {__version__}")
    sub = p.add_subparsers(dest="command", metavar="COMMAND")
    register_all(sub)
    return p


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = getattr(args, "handler", None)
    if handler is None:
        parser.print_help(sys.stderr)
        return 2
    return int(handler(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
