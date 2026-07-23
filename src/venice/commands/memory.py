"""`venice memory` -- inspect/manage the agent's persistent memory + tasks (#49).

`venice chat --memory` / `venice code --memory` let the agent keep durable notes and a
task checklist. Notes live in TWO tiers -- project (``<root>/.venice/memory/``, rides the
repo) and global (``~/.config/venice/memory/``, ``$VENICE_MEMORY_DIR`` overrides) -- while
tasks are project-only. This command is the read/manage surface; the store itself lives
in ``venice.commands._memory``.

Hygiene (CLAUDE.md): entry names are refused if secret-shaped at write time, so nothing
here can surface a credential; the store files are 0600.
"""
import sys

from . import _memory


def register(subparsers) -> None:
    p = subparsers.add_parser(
        "memory",
        help="List, show, or remove the agent's saved memory notes and tasks.",
        description=(
            "Inspect the persistent memory + task store `venice chat --memory` / "
            "`venice code --memory` write to. Notes have two tiers: project "
            "(<root>/.venice/memory) and global (~/.config/venice/memory, "
            "$VENICE_MEMORY_DIR overrides); tasks are project-only."
        ),
    )
    # Bare `venice memory` prints help (sub is optional), mirroring `venice sessions`.
    p.set_defaults(handler=_memory_help, memory_parser=p)
    sub = p.add_subparsers(dest="memory_command", metavar="ACTION")

    ls = sub.add_parser(
        "ls", aliases=["list"],
        help="List saved memory notes (metadata only).",
    )
    _add_scope(ls, "List only this tier (default: both).")
    ls.set_defaults(handler=_run_ls)

    show = sub.add_parser(
        "show", aliases=["cat"],
        help="Show one memory note (including its body).",
    )
    show.add_argument("name", help="Note name (see `venice memory ls`).")
    _add_scope(show, "Look in only this tier (default: project then global).")
    show.set_defaults(handler=_run_show)

    rm = sub.add_parser(
        "rm", aliases=["remove"], help="Delete a saved memory note.",
    )
    rm.add_argument("name")
    _add_scope(rm, "Tier to delete from (default: project).")
    rm.set_defaults(handler=_run_rm)

    tasks = sub.add_parser(
        "tasks", help="List the project's task checklist.",
    )
    tasks.add_argument(
        "--status", choices=list(_memory.TASK_STATUSES), default=None,
        help="Only tasks in this status.",
    )
    tasks.set_defaults(handler=_run_tasks)


def _add_scope(parser, help_text: str) -> None:
    parser.add_argument(
        "--scope", choices=list(_memory.SCOPES), default=None, help=help_text,
    )


def _memory_help(args) -> int:
    args.memory_parser.print_help(sys.stderr)
    return 2


def _run_ls(args) -> int:
    try:
        rows = _memory.list_entries(scope=args.scope)
    except _memory.MemStoreError as e:
        print(f"memory: {e}", file=sys.stderr)
        return 1
    if not rows:
        where = args.scope or "project/global"
        print(f"no saved memory notes ({where}). The agent writes them with "
              "`venice chat --memory` / `venice code --memory`.", file=sys.stderr)
        return 0
    for m in rows:
        desc = f"  {m['description']}" if m.get("description") else ""
        print(f"{m['name']}  [{m['scope']}/{m['type']}]  {m['updated']}{desc}")
    return 0


def _run_show(args) -> int:
    try:
        entry = _memory.read_entry(args.name, scope=args.scope)
    except _memory.MemStoreError as e:
        print(f"memory: {e}", file=sys.stderr)
        return 1
    if entry is None:
        print(f"memory: no note named {args.name!r}", file=sys.stderr)
        return 1
    print(f"name:    {entry['name']}")
    print(f"scope:   {entry['scope']}")
    print(f"type:    {entry['type']}")
    if entry.get("description"):
        print(f"description: {entry['description']}")
    print(f"created: {entry['created']}")
    print(f"updated: {entry['updated']}")
    print("---")
    print(entry.get("content", ""))
    return 0


def _run_rm(args) -> int:
    scope = args.scope or "project"
    try:
        removed = _memory.delete_entry(args.name, scope=scope)
    except _memory.MemStoreError as e:
        print(f"memory: {e}", file=sys.stderr)
        return 1
    if not removed:
        print(f"memory: no note named {args.name!r} in {scope}", file=sys.stderr)
        return 1
    print(f"removed memory note {args.name!r} ({scope}).", file=sys.stderr)
    return 0


def _run_tasks(args) -> int:
    try:
        tasks = _memory.list_tasks(status=args.status)
    except _memory.MemStoreError as e:
        print(f"memory: {e}", file=sys.stderr)
        return 1
    if not tasks:
        print("no tasks in this project.", file=sys.stderr)
        return 0
    for t in tasks:
        print(f"{t['id']}  [{t['status']}]  {t['text']}")
    return 0
