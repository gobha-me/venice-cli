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
    try:
        return int(handler(args) or 0)
    except KeyboardInterrupt:
        # The net under the per-command Ctrl-C handlers (#92). Each of those returns 130
        # long before the exception could reach here, so this only ever catches what they
        # miss -- and what they miss used to be a raw traceback. Deliberately silent: the
        # command that owns the interrupt has already printed its own notice, and this
        # must not print a second one over it. 130 is what CPython exited with anyway,
        # so no exit code changes; see the table in the README.
        print(file=sys.stderr)  # newline past a half-drawn prompt
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
