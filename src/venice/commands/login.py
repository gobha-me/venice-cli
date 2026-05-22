"""`venice login` -- interactive credential setup."""
import sys

from .. import auth


def register(subparsers) -> None:
    p = subparsers.add_parser(
        "login",
        help="Store your Venice API key (interactive, hidden input).",
        description=(
            "Read the Venice API key from a hidden prompt and store it at "
            "~/.config/venice/credentials with mode 0600. "
            "Set $VENICE_API_KEY to override without touching disk."
        ),
    )
    p.set_defaults(handler=_run)


def _run(args) -> int:
    try:
        auth.prompt_and_save()
        return 0
    except auth.AuthError as e:
        print(f"login failed: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\naborted.", file=sys.stderr)
        return 130
