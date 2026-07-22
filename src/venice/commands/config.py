"""venice config -- manage ~/.config/venice/config.json.

Two surfaces:
- MCP server registry (`add`/`list`/`remove`/`show`), modeled on `claude mcp add`,
  which the `venice chat --mcp` external-MCP client (#21) will load.
- Default flag values (`get`/`set`/`unset`) so users stop repeating flags (#17).

The API key is never stored here; it stays in ~/.config/venice/credentials.
This is the repo's first nested subparser: bare `venice config` prints help and
exits 2 (the intermediate parser keeps its own default handler).
"""
import json
import sys

from .. import userconfig


def register(subparsers) -> None:
    p = subparsers.add_parser(
        "config",
        help="Manage persistent config (MCP server registry + default flags).",
        description=(
            "Read/write ~/.config/venice/config.json: an MCP server registry "
            "(loaded by `venice chat --mcp`) and default flag values so you stop "
            "repeating --model / -o / --yes / --max-spend. Precedence for a flag "
            "is CLI > env > config > built-in default. The API key is NEVER stored "
            "here -- it stays in ~/.config/venice/credentials."
        ),
    )
    # Bare `venice config` falls through to this help handler (csub is optional).
    p.set_defaults(handler=_config_help, config_parser=p)
    csub = p.add_subparsers(dest="config_command", metavar="ACTION")

    add = csub.add_parser(
        "add",
        help="Register an MCP server (stdio command or http/sse URL).",
        description=(
            "Register an MCP server by name. Give either --command (a stdio "
            "server, e.g. `venice config add venice --command venice --arg "
            "mcp-serve`) or --url (an http/sse server). An --env or --header "
            "value may contain `@secret:<name>` to pull the value from the "
            "0600 secret store at attach time (e.g. --header "
            "'Authorization: Bearer @secret:cluster') instead of storing the "
            "token in plaintext; add the secret with `venice secret set <name>`."
        ),
    )
    add.add_argument("name", help="Registry name, e.g. 'venice' or 'filesystem'.")
    # dest must NOT be 'command' -- the top-level subparser already owns that dest.
    add.add_argument("--command", dest="server_command", metavar="CMD",
                     help="Executable for a stdio server (e.g. venice, npx).")
    add.add_argument("--arg", dest="arg", action="append", default=[], metavar="ARG",
                     help="Argument for the stdio command (repeatable).")
    add.add_argument("--env", dest="env", action="append", default=[], metavar="K=V",
                     help="Environment variable for a stdio server (repeatable); "
                          "the value may contain @secret:<name>.")
    add.add_argument("--url", dest="url", metavar="URL",
                     help="Endpoint for an http/sse server.")
    add.add_argument("--type", dest="server_type", choices=("http", "sse"),
                     default="http", help="Transport for a --url server (default: http).")
    add.add_argument("--header", dest="header", action="append", default=[], metavar="K: V",
                     help="HTTP header for a --url server (repeatable); the "
                          "value may contain @secret:<name>.")
    add.add_argument("--force", action="store_true",
                     help="Overwrite an existing entry of the same name.")
    add.set_defaults(handler=_run_add)

    lst = csub.add_parser("list", help="List registered MCP servers.")
    lst.add_argument("--json", action="store_true",
                     help="Emit the raw mcpServers map as JSON.")
    lst.set_defaults(handler=_run_list)

    rm = csub.add_parser("remove", help="Remove a registered MCP server.")
    rm.add_argument("name")
    rm.set_defaults(handler=_run_remove)

    show = csub.add_parser("show", help="Print the whole config, or one server entry.")
    show.add_argument("name", nargs="?", help="Show just this MCP server entry.")
    show.add_argument("--json", action="store_true",
                      help="(default output is already JSON)")
    show.set_defaults(handler=_run_show)

    get = csub.add_parser("get", help="Print a config value by dotted key.")
    get.add_argument("key", help="Dotted key, e.g. defaults.chat.model.")
    get.set_defaults(handler=_run_get)

    st = csub.add_parser("set", help="Set a config value by dotted key.")
    st.add_argument("key", help="Dotted key, e.g. defaults.chat.model.")
    st.add_argument("value", help="Value (parsed as JSON when possible, else a string).")
    st.set_defaults(handler=_run_set)

    un = csub.add_parser("unset", help="Remove a config value by dotted key.")
    un.add_argument("key")
    un.set_defaults(handler=_run_unset)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _config_help(args) -> int:
    args.config_parser.print_help(sys.stderr)
    return 2


def _parse_pairs(items, sep, label):
    """Parse ['K<sep>V', ...] into a dict. Returns (dict, error_message_or_None)."""
    out = {}
    for item in items:
        if sep not in item:
            return None, f"bad {label} {item!r} (expected K{sep}V)"
        k, v = item.split(sep, 1)
        out[k.strip()] = v.strip()
    return out, None


def _summary(entry) -> str:
    if not isinstance(entry, dict):
        return repr(entry)
    if "command" in entry:
        parts = [entry["command"], *entry.get("args", [])]
        return "stdio: " + " ".join(str(x) for x in parts)
    if "url" in entry:
        return f"{entry.get('type', 'http')}: {entry['url']}"
    return "(unrecognized entry)"


def _coerce_value(s: str):
    """JSON-parse a `set` value so numbers/bools/null get real types; fall back
    to the raw string for barewords like a model id."""
    try:
        return json.loads(s)
    except ValueError:
        return s


# --------------------------------------------------------------------------- #
# MCP registry actions (#13)
# --------------------------------------------------------------------------- #
def _run_add(args) -> int:
    has_cmd = bool(args.server_command)
    has_url = bool(args.url)
    if has_cmd == has_url:
        print("config add: give exactly one of --command (stdio) or --url (http/sse).",
              file=sys.stderr)
        return 2

    if has_cmd:
        env, err = _parse_pairs(args.env, "=", "--env")
        if err:
            print(f"config add: {err}", file=sys.stderr)
            return 2
        entry = {"command": args.server_command}
        if args.arg:
            entry["args"] = list(args.arg)
        if env:
            entry["env"] = env
    else:
        headers, err = _parse_pairs(args.header, ":", "--header")
        if err:
            print(f"config add: {err}", file=sys.stderr)
            return 2
        entry = {"type": args.server_type, "url": args.url}
        if headers:
            entry["headers"] = headers

    try:
        doc = userconfig.load_config_for_write()
    except userconfig.ConfigError as e:
        print(f"config add: {e}", file=sys.stderr)
        return 2

    if userconfig.mcp_get(doc, args.name) is not None and not args.force:
        print(f"config add: {args.name!r} already exists (use --force to overwrite).",
              file=sys.stderr)
        return 2

    userconfig.mcp_add(doc, args.name, entry)
    try:
        path = userconfig.save_config(doc)
    except OSError as e:
        print(f"config add: cannot write config: {e}", file=sys.stderr)
        return 9
    print(f"added MCP server {args.name!r} to {path}")
    return 0


def _run_list(args) -> int:
    doc = userconfig.load_config()
    servers = userconfig.mcp_map(doc)
    if args.json:
        print(json.dumps(servers, indent=2, sort_keys=True))
        return 0
    if not servers:
        print("(no MCP servers registered)")
        return 0
    for name in sorted(servers):
        print(f"{name}\t{_summary(servers[name])}")
    return 0


def _run_remove(args) -> int:
    try:
        doc = userconfig.load_config_for_write()
    except userconfig.ConfigError as e:
        print(f"config remove: {e}", file=sys.stderr)
        return 2
    if not userconfig.mcp_remove(doc, args.name):
        available = sorted(userconfig.mcp_map(doc))
        hint = f" (available: {', '.join(available)})" if available else ""
        print(f"config remove: no MCP server named {args.name!r}{hint}", file=sys.stderr)
        return 2
    try:
        userconfig.save_config(doc)
    except OSError as e:
        print(f"config remove: cannot write config: {e}", file=sys.stderr)
        return 9
    print(f"removed MCP server {args.name!r}")
    return 0


def _run_show(args) -> int:
    doc = userconfig.load_config()
    if args.name:
        entry = userconfig.mcp_get(doc, args.name)
        if entry is None:
            print(f"config show: no MCP server named {args.name!r}", file=sys.stderr)
            return 2
        print(json.dumps(entry, indent=2, sort_keys=True))
        return 0
    print(json.dumps(doc, indent=2, sort_keys=True))
    return 0


# --------------------------------------------------------------------------- #
# Default-flag actions (#17)
# --------------------------------------------------------------------------- #
def _run_get(args) -> int:
    doc = userconfig.load_config()
    try:
        val = userconfig.get_value(doc, args.key)
    except KeyError:
        print(f"config get: {args.key!r} is not set", file=sys.stderr)
        return 2
    if isinstance(val, str):
        print(val)
    else:
        print(json.dumps(val, indent=2, sort_keys=True))
    return 0


def _run_set(args) -> int:
    value = _coerce_value(args.value)
    try:
        doc = userconfig.load_config_for_write()
        userconfig.set_value(doc, args.key, value)
    except userconfig.ConfigError as e:
        print(f"config set: {e}", file=sys.stderr)
        return 2
    try:
        userconfig.save_config(doc)
    except OSError as e:
        print(f"config set: cannot write config: {e}", file=sys.stderr)
        return 9
    print(f"set {args.key} = {json.dumps(value)}")
    return 0


def _run_unset(args) -> int:
    try:
        doc = userconfig.load_config_for_write()
    except userconfig.ConfigError as e:
        print(f"config unset: {e}", file=sys.stderr)
        return 2
    if not userconfig.unset_value(doc, args.key):
        print(f"config unset: {args.key!r} is not set", file=sys.stderr)
        return 2
    try:
        userconfig.save_config(doc)
    except OSError as e:
        print(f"config unset: cannot write config: {e}", file=sys.stderr)
        return 9
    print(f"unset {args.key}")
    return 0
