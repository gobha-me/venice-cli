"""`venice secret` -- manage the local 0600 named-secret store (#43).

Named secrets (the embed-backend key, later MCP/cluster tokens) live in
`~/.config/venice/secrets.json` (mode 0600), separate from the single Venice key in
`credentials` (`venice login`) and from plaintext `config.json`. This command is the
CRUD surface; the store itself lives in `venice.auth`.

Hygiene (CLAUDE.md): a value is only ever read via a hidden getpass prompt and is
never printed back. `ls` shows names + character counts, never values -- there is
deliberately no command that prints a secret.
"""
import sys

from .. import auth


def register(subparsers) -> None:
    p = subparsers.add_parser(
        "secret",
        help="Manage named secrets in the local 0600 secret store.",
        description=(
            "Store/list/remove named secrets in ~/.config/venice/secrets.json "
            "(mode 0600) -- e.g. the embed-backend key ('embed'). Values are read "
            "from a hidden prompt and are NEVER printed back; `ls` shows lengths "
            "only. The main Venice API key is separate (`venice login`). A name "
            "with a canonical env var (embed -> $VENICE_EMBED_API_KEY) still lets "
            "the env value override the stored one."
        ),
    )
    # Bare `venice secret` falls through to this help handler (ssub is optional).
    p.set_defaults(handler=_secret_help, secret_parser=p)
    ssub = p.add_subparsers(dest="secret_command", metavar="ACTION")

    st = ssub.add_parser(
        "set",
        help="Store a named secret (hidden prompt; never on argv).",
        description="Prompt (hidden) for a value and store it under <name>.",
    )
    st.add_argument("name", help="Secret name, e.g. 'embed'.")
    st.set_defaults(handler=_run_set)

    ls = ssub.add_parser(
        "ls", aliases=["list"],
        help="List secret names and lengths (never the values).",
    )
    ls.set_defaults(handler=_run_ls)

    rm = ssub.add_parser(
        "rm", aliases=["remove"], help="Delete a named secret.",
    )
    rm.add_argument("name")
    rm.set_defaults(handler=_run_rm)


def _secret_help(args) -> int:
    args.secret_parser.print_help(sys.stderr)
    return 2


def _run_set(args) -> int:
    try:
        auth.prompt_and_save_secret(args.name)
        return 0
    except auth.AuthError as e:
        print(f"secret: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\naborted.", file=sys.stderr)
        return 130


def _run_ls(args) -> int:
    entries = auth.list_secrets()
    if not entries:
        print("no secrets stored (add one with `venice secret set <name>`).",
              file=sys.stderr)
        return 0
    for name, length in entries:
        print(f"{name}  ({length} chars)")
    return 0


def _run_rm(args) -> int:
    try:
        removed = auth.delete_secret(args.name)
    except auth.AuthError as e:
        print(f"secret: {e}", file=sys.stderr)
        return 1
    if not removed:
        print(f"secret: no secret named {args.name!r}", file=sys.stderr)
        return 1
    print(f"removed secret {args.name!r}.", file=sys.stderr)
    return 0
