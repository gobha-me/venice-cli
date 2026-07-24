"""`venice sessions` -- list/show/remove/steer saved chat & code sessions (#47, #78).

`venice chat`/`venice code` auto-save each REPL session (id + settings + usage +
transcript) under ``~/.config/venice/sessions/`` (``$VENICE_SESSIONS_DIR`` overrides).
This command is the read/manage surface; the store itself lives in
``venice.commands._session``. Resume a listed session with
``venice chat --resume <id>`` (or ``--continue`` for the most recent).

Hygiene (CLAUDE.md): envelopes hold only messages + settings + usage, never the API
key. ``show`` prints metadata and message roles/counts -- it does not dump raw content
by default and there is no command that could surface a stored credential.
"""
import json
import sys

from . import _mailbox, _session


def register(subparsers) -> None:
    p = subparsers.add_parser(
        "sessions",
        help="List, show, steer, or remove saved chat/code sessions.",
        description=(
            "Manage auto-saved `venice chat`/`venice code` sessions in "
            "~/.config/venice/sessions/ ($VENICE_SESSIONS_DIR overrides). Resume "
            "one with `venice chat --resume <id>` or `--continue`. Sessions carry "
            "their model, settings, usage, and transcript; the API key is never stored."
        ),
    )
    # Bare `venice sessions` prints help (ssub is optional), mirroring `venice secret`.
    p.set_defaults(handler=_sessions_help, sessions_parser=p)
    ssub = p.add_subparsers(dest="sessions_command", metavar="ACTION")

    ls = ssub.add_parser(
        "ls", aliases=["list"],
        help="List saved sessions (newest first).",
    )
    ls.add_argument(
        "--json", action="store_true",
        help="Emit the listing as JSON (adds each session's pending-steer count).",
    )
    ls.set_defaults(handler=_run_ls)

    show = ssub.add_parser(
        "show", aliases=["cat"],
        help="Show one session's settings and message summary.",
    )
    show.add_argument("id", help="Session id (see `venice sessions ls`).")
    show.set_defaults(handler=_run_show)

    rm = ssub.add_parser(
        "rm", aliases=["remove"], help="Delete a saved session.",
    )
    rm.add_argument("id")
    rm.set_defaults(handler=_run_rm)

    # #78 mid-run steering: queue a message into a session's file mailbox. A running
    # `venice code` agent drains it at its next checkpoint and consumes it as a user
    # turn -- no kill, no lost work. `latest` targets the most recent code session.
    send = ssub.add_parser(
        "send",
        help="Queue a steering message for a running/saved session (mid-run steering).",
        description=(
            "Write a message into a session's mailbox; a running `venice code` agent "
            "drains it at its next checkpoint and consumes it as if you had typed it "
            "interactively. TARGET is a session id or `latest` (the most recent code "
            "session). Pass `-` as MESSAGE to read it from stdin. Only saved "
            "(non-ephemeral) sessions are steerable; targeting is by recency, not "
            "liveness, so a message to a finished session waits for the next --resume."
        ),
    )
    send.add_argument("target", help="Session id, or `latest` for the newest code session.")
    send.add_argument("message", help="The steering message (or `-` to read stdin).")
    send.set_defaults(handler=_run_send)


def _sessions_help(args) -> int:
    args.sessions_parser.print_help(sys.stderr)
    return 2


def _run_ls(args) -> int:
    rows = _session.list_sessions()
    if getattr(args, "json", False):
        out = [
            {
                "id": sid, "command": command, "updated": updated,
                "messages": n_msgs, "model": model,
                "pending": _mailbox.pending(sid),  # queued steers (#78)
            }
            for sid, command, updated, n_msgs, model in rows
        ]
        json.dump(out, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0
    if not rows:
        print(
            "no saved sessions yet (start `venice chat` or `venice code`; "
            "auto-save is on unless --ephemeral).",
            file=sys.stderr,
        )
        return 0
    for sid, command, updated, n_msgs, model in rows:
        model_s = f"  {model}" if model else ""
        n_pending = _mailbox.pending(sid)  # #78: flag queued steers inline
        pending_s = f"  ({n_pending} pending)" if n_pending else ""
        print(f"{sid}  [{command}]  {updated}  {n_msgs} msgs{model_s}{pending_s}")
    return 0


def _run_show(args) -> int:
    try:
        # command is only a fallback for a bare-list import; a store id carries its own.
        sess = _session.load(args.id, "chat")
    except _session.SessionError as e:
        print(f"sessions: {e}", file=sys.stderr)
        return 1
    print(f"id:      {sess.id}")
    print(f"command: {sess.command}")
    print(f"created: {sess.created}")
    print(f"updated: {sess.updated}")
    if sess.model:
        print(f"model:   {sess.model}")
    if sess.root:
        print(f"root:    {sess.root}")
    if sess.max_tool_calls is not None:
        print(f"max-tool-calls: {sess.max_tool_calls}")
    if sess.gen_kwargs:
        print(f"gen_kwargs: {sess.gen_kwargs}")
    if sess.usage:
        print(f"usage:   {sess.usage}")
    print(f"messages: {len(sess.messages)}")
    roles: dict = {}
    for m in sess.messages:
        role = m.get("role", "?") if isinstance(m, dict) else "?"
        roles[role] = roles.get(role, 0) + 1
    if roles:
        summary = ", ".join(f"{r}={c}" for r, c in sorted(roles.items()))
        print(f"  by role: {summary}")
    n_pending = _mailbox.pending(sess.id)  # #78: queued steering messages
    if n_pending:
        print(f"pending steers: {n_pending}")
    return 0


def _run_send(args) -> int:
    """Queue a steering message (#78) into a session's mailbox.

    Resolves `latest` to the most recent code session; validates a bare id against
    the store (rejecting traversal); reads `-` from stdin. Prints the queued count.
    """
    target = args.target
    if target == "latest":
        sess = _session.most_recent("code")
        if sess is None:
            print("sessions: no code session to steer (start `venice code`).",
                  file=sys.stderr)
            return 1
        sid = sess.id
    else:
        try:
            path = _session._resolve_zone_path(target)  # validates id (no traversal)
        except _session.SessionError as e:
            print(f"sessions: {e}", file=sys.stderr)
            return 1
        if not path.is_file():
            print(f"sessions: no session named {target!r}", file=sys.stderr)
            return 1
        sid = target

    message = args.message
    if message == "-":
        message = sys.stdin.read()
    message = (message or "").strip()
    if not message:
        print("sessions: empty steering message", file=sys.stderr)
        return 2

    try:
        _mailbox.deposit(sid, message)
    except OSError as e:
        print(f"sessions: could not queue message: {e}", file=sys.stderr)
        return 1
    print(f"queued steering message for session {sid} "
          f"({_mailbox.pending(sid)} pending)", file=sys.stderr)
    return 0


def _run_rm(args) -> int:
    try:
        removed = _session.delete(args.id)
    except _session.SessionError as e:
        print(f"sessions: {e}", file=sys.stderr)
        return 1
    if not removed:
        print(f"sessions: no session named {args.id!r}", file=sys.stderr)
        return 1
    print(f"removed session {args.id!r}.", file=sys.stderr)
    return 0
