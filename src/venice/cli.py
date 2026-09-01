"""Top-level argparse dispatcher."""
import argparse
import os
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
        try:
            code = int(handler(args) or 0)
        except KeyboardInterrupt:
            # The net under the per-command Ctrl-C handlers (#92). Each of those
            # returns 130 long before the exception could reach here, so this only
            # ever catches what they miss -- and what they miss used to be a raw
            # traceback. Deliberately silent: the command that owns the interrupt
            # has already printed its own notice, and this must not print a second
            # one over it. 130 is what CPython exited with anyway, so no exit code
            # changes; see the table in the README.
            print(file=sys.stderr)  # newline past a half-drawn prompt
            code = 130

        # A pipe can accept every write above, then close before CPython's buffered
        # stdout is flushed during interpreter shutdown. Flush while the dispatcher
        # can still turn that condition into the documented producer exit (#94).
        sys.stdout.flush()
        return code
    except BrokenPipeError:
        # Replacing the stream keeps CPython's shutdown flush from reporting the
        # same EPIPE a second time as "Exception ignored in ...". Do not print a
        # notice: the downstream reader deliberately stopped consuming output.
        sys.stdout = open(os.devnull, "w", encoding="utf-8")
        return 141  # 128 + SIGPIPE


if __name__ == "__main__":
    raise SystemExit(main())
