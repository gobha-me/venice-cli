"""Built-in coding toolset for `venice code` (vcoder, issue #29).

The self-contained-agent loop (#15) plus these tools turn a Venice text model into
a coding agent: it reads, searches, edits, and runs commands in a project tree. The
tools are **built-in** (not an external MCP filesystem/exec server) so path-scoping
and the confirm gate stay under this CLI's control -- the deliberate epic-#25
decision. `venice chat --mcp` (#21) remains the extension point for extra tools.

Import discipline (mirrors `_agent`/`_mcp`): stdlib only, and **no `mcp` SDK** -- the
coding toolset needs only the `[openai]` extra (for the loop) and runs on the 3.9
floor. The security-critical path helpers are reused from `commands._index`
(`resolves_inside`, `is_secret_path`, `read_text`, `walk_files`) so there is one
source of truth for "is this path inside the project?" and "is it secret-shaped?".

Safety model:
- **Path sandbox.** Every fs path is joined to `root`, ``realpath``-resolved, and
  gated with :func:`_index.resolves_inside`; a path that escapes the root, names a
  secret-shaped file, or lives under a protected dir (``.git``/``.venice``/vendor) is
  refused. `root` is realpath-resolved once by :func:`code_tools`.
- **Confirm gate.** Mutating tools (``write_file``/``edit_file``/``run``) are
  ``paid=True``: their ``invoke`` returns ``{"status": "confirmation_required"}`` when
  ``confirm`` is false (no side effect, no round-trip) and performs the action when
  it is true -- exactly the gate the paid venice tools and side-effecting MCP tools
  use, so ``--yes``/autonomous mode bypasses it uniformly. Loop-controlled kwargs
  (``confirm``/``max_spend``/``output_dir``) are stripped defensively and never
  advertised in a tool's JSON schema.
- **exec honesty.** ``run`` executes ``/bin/sh -c`` with cwd forced to `root`, a
  timeout, size-capped captured output, and the Venice API keys scrubbed from the
  child env. A shell command can still read/write **outside** the root (``cat ../x``);
  exec's boundary is the confirm gate + cwd + timeout + env-scrub, **not** path
  containment -- which is why it is always gated. ``git`` exposes only read-only
  subcommands freely; mutations go through the (gated) ``run`` tool.
"""
from __future__ import annotations

import fnmatch
import os
import re
import subprocess
from pathlib import Path
from typing import List

from . import _agent, _index, _mcp

# --------------------------------------------------------------------------- #
# Limits + constants
# --------------------------------------------------------------------------- #
MAX_READ_BYTES = _index.MAX_FILE_BYTES        # reuse the indexer's oversize cap
MAX_OUTPUT_CHARS = 20_000                      # cap per stdout/stderr stream
MAX_GREP_MATCHES = 200
MAX_GREP_FILES = 5_000
DEFAULT_EXEC_TIMEOUT = 120                     # seconds

# Secrets never inherited into an exec'd child (CLAUDE.md credential hygiene).
_SECRET_ENV = ("VENICE_API_KEY", "VENICE_EMBED_API_KEY")

# Loop-controlled kwargs the model must never supply (mirrors _agent._CONTROLLED).
_CONTROLLED = ("confirm", "max_spend", "output_dir")


def _clean(arguments) -> dict:
    if not isinstance(arguments, dict):
        return {}
    return {k: v for k, v in arguments.items() if k not in _CONTROLLED}


def _err(message: str) -> dict:
    return {"status": "error", "message": message}


def _ok(**kw) -> dict:
    return {"status": "ok", **kw}


def _confirm(message: str) -> dict:
    return {"status": "confirmation_required", "message": message}


class _PathError(Exception):
    """A path failed the sandbox / denylist checks. Its message is printable."""


# --------------------------------------------------------------------------- #
# Path sandbox
# --------------------------------------------------------------------------- #
def _safe_path(root: str, arg, *, must_exist: bool = False):
    """Resolve `arg` (relative to `root`) and enforce the sandbox + denylists.

    Returns ``(real_abspath, rel_posix)``. Raises :class:`_PathError` if the path
    escapes `root`, is secret-shaped, lives under a protected dir, or (when
    `must_exist`) does not exist. `root` must already be realpath-resolved.
    """
    if arg is None or not str(arg).strip():
        raise _PathError("path is required")
    joined = os.path.normpath(os.path.join(root, str(arg)))
    real = os.path.realpath(joined)
    if not _index.resolves_inside(Path(real), Path(root)):
        raise _PathError(f"path escapes the project root: {arg}")
    rel = Path(os.path.relpath(real, root)).as_posix()
    if _index.is_secret_path(rel) or _index.is_protected_dir_path(rel):
        raise _PathError(f"path is in a protected location (secret/.git/.venice): {arg}")
    if must_exist and not os.path.exists(real):
        raise _PathError(f"no such file or directory: {arg}")
    return real, rel


def _atomic_write(target: str, text: str) -> None:
    """Write `text` to `target` atomically (tmp in the same dir + fsync + replace).

    Normal file mode (umask-respecting) -- these are source files, not the 0600
    secrets `_index.save_store` writes. Creates parent dirs (already inside root).
    """
    parent = os.path.dirname(target) or "."
    os.makedirs(parent, exist_ok=True)
    tmp = target + ".venice-tmp"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    os.replace(tmp, target)


# --------------------------------------------------------------------------- #
# Tool implementations (print-free; return JSON-serializable dicts)
# --------------------------------------------------------------------------- #
def read_file(root: str, path, *, offset=None, limit=None) -> dict:
    try:
        real, rel = _safe_path(root, path, must_exist=True)
    except _PathError as e:
        return _err(str(e))
    if os.path.isdir(real):
        return _err(f"{rel} is a directory; use list_dir")
    data, text = _index.read_text(Path(real))
    if data is None:
        return _err(f"cannot read {rel}")
    if len(data) > MAX_READ_BYTES:
        return _err(f"{rel} is too large ({len(data)} bytes > {MAX_READ_BYTES})")
    if text is None:
        return _err(f"{rel} is binary or not UTF-8; not shown")
    lines = text.splitlines()
    total = len(lines)
    start = max(int(offset) - 1, 0) if offset else 0
    end = (start + int(limit)) if limit else total
    sel = lines[start:end]
    return _ok(
        path=rel,
        start=start + 1 if sel else 0,
        end=start + len(sel),
        total_lines=total,
        content="\n".join(sel),
        truncated=(start > 0 or end < total),
    )


def list_dir(root: str, *, path=".") -> dict:
    try:
        real, rel = _safe_path(root, path or ".", must_exist=True)
    except _PathError as e:
        return _err(str(e))
    if not os.path.isdir(real):
        return _err(f"{rel} is not a directory")
    entries = []
    try:
        names = sorted(os.listdir(real))
    except OSError as e:
        return _err(f"cannot list {rel}: {e}")
    for name in names:
        if _index.is_secret_name(name):
            continue  # never reveal secret-shaped files
        full = os.path.join(real, name)
        isdir = os.path.isdir(full)
        entry = {"name": name, "type": "dir" if isdir else "file"}
        if not isdir:
            try:
                entry["size"] = os.path.getsize(full)
            except OSError:
                pass
        entries.append(entry)
    return _ok(path=rel, entries=entries, count=len(entries))


def grep_files(
    root: str, pattern, *, path=None, glob=None, ignore_case=False,
    max_matches: int = MAX_GREP_MATCHES,
) -> dict:
    if not pattern or not str(pattern).strip():
        return _err("pattern is required")
    try:
        rx = re.compile(str(pattern), re.IGNORECASE if ignore_case else 0)
    except re.error as e:
        return _err(f"invalid regex: {e}")

    single_file = None
    walk_root = root
    if path:
        try:
            real, _rel = _safe_path(root, path, must_exist=True)
        except _PathError as e:
            return _err(str(e))
        if os.path.isfile(real):
            single_file = Path(real)
        else:
            walk_root = real

    cap = max(1, int(max_matches or MAX_GREP_MATCHES))
    matches: List[dict] = []
    truncated = False
    scanned = 0
    iterator = [single_file] if single_file else _index.walk_files(Path(walk_root))
    for fp in iterator:
        scanned += 1
        if scanned > MAX_GREP_FILES:
            truncated = True
            break
        rel = Path(os.path.relpath(str(fp), root)).as_posix()
        if glob and not (fnmatch.fnmatch(rel, glob) or fnmatch.fnmatch(fp.name, glob)):
            continue
        _data, text = _index.read_text(fp)
        if text is None:
            continue
        for i, line in enumerate(text.splitlines(), 1):
            if rx.search(line):
                matches.append({"path": rel, "line": i, "text": line[:400]})
                if len(matches) >= cap:
                    truncated = True
                    break
        if truncated:
            break
    return _ok(matches=matches, count=len(matches), truncated=truncated)


def write_file(root: str, path, content, *, confirm: bool = False) -> dict:
    if content is None:
        return _err("content is required")
    try:
        real, rel = _safe_path(root, path)
    except _PathError as e:
        return _err(str(e))
    if os.path.isdir(real):
        return _err(f"{rel} is a directory")
    existed = os.path.exists(real)
    if not confirm:
        verb = "overwrite" if existed else "create"
        return _confirm(f"write_file will {verb} {rel} ({len(str(content))} chars)")
    try:
        _atomic_write(real, str(content))
    except OSError as e:
        return _err(f"write failed: {e}")
    return _ok(path=rel, action="overwrote" if existed else "created",
               bytes=len(str(content).encode("utf-8")))


def edit_file(root: str, path, old, new, *, confirm: bool = False) -> dict:
    if old is None or new is None:
        return _err("both old and new are required")
    if old == "":
        return _err("old must be a non-empty string")
    try:
        real, rel = _safe_path(root, path, must_exist=True)
    except _PathError as e:
        return _err(str(e))
    if os.path.isdir(real):
        return _err(f"{rel} is a directory")
    _data, text = _index.read_text(Path(real))
    if text is None:
        return _err(f"{rel} is binary or unreadable")
    occ = text.count(str(old))
    if occ == 0:
        return _err("old string not found in file")
    if occ > 1:
        return _err(f"old string is not unique ({occ} occurrences); add surrounding context")
    if not confirm:
        return _confirm(f"edit_file will replace 1 occurrence in {rel}")
    try:
        _atomic_write(real, text.replace(str(old), str(new), 1))
    except OSError as e:
        return _err(f"write failed: {e}")
    return _ok(path=rel, action="edited")


def _scrubbed_env() -> dict:
    return {k: v for k, v in os.environ.items() if k not in _SECRET_ENV}


def run_cmd(root: str, command, *, timeout=None, exec_timeout: int = DEFAULT_EXEC_TIMEOUT,
            confirm: bool = False) -> dict:
    if not command or not str(command).strip():
        return _err("command is required")
    if not confirm:
        return _confirm(f"run will execute in {root}:\n    {command}")
    try:
        t = int(timeout) if timeout else int(exec_timeout)
    except (TypeError, ValueError):
        t = int(exec_timeout)
    try:
        proc = subprocess.run(
            ["/bin/sh", "-c", str(command)], cwd=root, capture_output=True,
            text=True, timeout=t, env=_scrubbed_env(),
        )
    except subprocess.TimeoutExpired:
        return _err(f"command timed out after {t}s")
    except OSError as e:
        return _err(f"could not run command: {e}")
    out, errout = proc.stdout or "", proc.stderr or ""
    return _ok(
        exit_code=proc.returncode,
        stdout=out[:MAX_OUTPUT_CHARS],
        stderr=errout[:MAX_OUTPUT_CHARS],
        truncated=(len(out) > MAX_OUTPUT_CHARS or len(errout) > MAX_OUTPUT_CHARS),
    )


_GIT_READONLY = frozenset({
    "status", "diff", "log", "show", "branch", "ls-files", "blame", "remote",
    "rev-parse", "describe", "shortlog",
})


def git_cmd(root: str, subcommand, *, args=None,
            exec_timeout: int = DEFAULT_EXEC_TIMEOUT) -> dict:
    sub = str(subcommand or "").strip()
    if sub not in _GIT_READONLY:
        return _err(
            "git: only read-only subcommands are allowed here "
            f"({', '.join(sorted(_GIT_READONLY))}); use the run tool "
            "(which confirms) for mutations like add/commit"
        )
    argv = ["git", sub]
    if args:
        if not isinstance(args, list):
            return _err("args must be a list of strings")
        for a in args:
            if not isinstance(a, (str, int, float)):
                return _err("each arg must be a string")
            argv.append(str(a))
    try:
        proc = subprocess.run(
            argv, cwd=root, capture_output=True, text=True,
            timeout=int(exec_timeout), env=_scrubbed_env(),
        )
    except FileNotFoundError:
        return _err("git is not installed")
    except subprocess.TimeoutExpired:
        return _err("git command timed out")
    except OSError as e:
        return _err(f"git failed: {e}")
    out, errout = proc.stdout or "", proc.stderr or ""
    return _ok(
        exit_code=proc.returncode,
        stdout=out[:MAX_OUTPUT_CHARS],
        stderr=errout[:MAX_OUTPUT_CHARS],
        truncated=(len(out) > MAX_OUTPUT_CHARS),
    )


# --------------------------------------------------------------------------- #
# JSON schemas (literals; confirm/max_spend/output_dir deliberately absent)
# --------------------------------------------------------------------------- #
def _p(typ, desc=None):
    d = {"type": typ}
    if desc:
        d["description"] = desc
    return d


def _obj(props, required=None):
    s = {"type": "object", "properties": props}
    if required:
        s["required"] = required
    return s


_READ_SCHEMA = _obj(
    {
        "path": _p("string", "File path, relative to the project root."),
        "offset": _p("integer", "1-based first line to return (default: 1)."),
        "limit": _p("integer", "Maximum number of lines to return."),
    },
    ["path"],
)
_LIST_SCHEMA = _obj(
    {"path": _p("string", "Directory path relative to the root (default: '.').")},
)
_GREP_SCHEMA = _obj(
    {
        "pattern": _p("string", "Python regular expression to search for."),
        "path": _p("string", "Limit the search to this file or subdirectory."),
        "glob": _p("string", "Only search files whose path matches this glob."),
        "ignore_case": _p("boolean", "Case-insensitive match."),
        "max_matches": _p("integer", f"Cap results (default {MAX_GREP_MATCHES})."),
    },
    ["pattern"],
)
_WRITE_SCHEMA = _obj(
    {
        "path": _p("string", "File path relative to the root (created if new)."),
        "content": _p("string", "Full new contents of the file."),
    },
    ["path", "content"],
)
_EDIT_SCHEMA = _obj(
    {
        "path": _p("string", "File to edit, relative to the root."),
        "old": _p("string", "Exact text to replace; must occur exactly once."),
        "new": _p("string", "Replacement text."),
    },
    ["path", "old", "new"],
)
_RUN_SCHEMA = _obj(
    {
        "command": _p("string", "Shell command to run (via /bin/sh -c) in the root."),
        "timeout": _p("integer", "Timeout in seconds for this command."),
    },
    ["command"],
)
_GIT_SCHEMA = _obj(
    {
        "subcommand": _p("string", "Read-only git subcommand (status/diff/log/show/...)."),
        "args": {"type": "array", "items": {"type": "string"},
                 "description": "Extra arguments for the subcommand."},
    },
    ["subcommand"],
)
_SEARCH_SCHEMA = _obj(
    {
        "query": _p("string", "Natural-language description of the code to find."),
        "k": _p("integer", "Number of results (default 8)."),
    },
    ["query"],
)


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
def code_tools(
    root: str,
    client=None,
    *,
    exec_timeout: int = DEFAULT_EXEC_TIMEOUT,
    include_search: bool = False,
) -> List[_agent.Tool]:
    """Build the coding tools bound to a realpath-resolved project `root`.

    Read-only tools (`read_file`/`list_dir`/`grep`/`git`) are free; mutating tools
    (`write_file`/`edit_file`/`run`) are ``paid=True`` and route through the confirm
    gate. `project_search` (reusing the free `_mcp.search_tool`) is added only when
    `include_search` and a `client` are supplied and a `.venice` index exists.
    """
    root = os.path.realpath(root)

    def free(fn):
        def invoke(arguments, *, confirm: bool = False):
            return fn(root, **_clean(arguments))
        return invoke

    def paid(fn):
        def invoke(arguments, *, confirm: bool = False):
            return fn(root, confirm=confirm, **_clean(arguments))
        return invoke

    def run_invoke(arguments, *, confirm: bool = False):
        return run_cmd(root, confirm=confirm, exec_timeout=exec_timeout,
                       **_clean(arguments))

    def git_invoke(arguments, *, confirm: bool = False):
        return git_cmd(root, exec_timeout=exec_timeout, **_clean(arguments))

    tools = [
        _agent.Tool("read_file",
                    "Read a UTF-8 text file inside the project root and return its "
                    "lines. Use before editing. Read-only.",
                    _READ_SCHEMA, free(read_file), paid=False),
        _agent.Tool("list_dir",
                    "List the entries of a directory inside the project root "
                    "(secret-shaped files are hidden). Read-only.",
                    _LIST_SCHEMA, free(list_dir), paid=False),
        _agent.Tool("grep",
                    "Search project files for a regular expression and return "
                    "matching path:line:text. Read-only.",
                    _GREP_SCHEMA, free(grep_files), paid=False),
        _agent.Tool("write_file",
                    "Create or overwrite a file inside the project root with new "
                    "contents. Mutating -- requires confirmation.",
                    _WRITE_SCHEMA, paid(write_file), paid=True),
        _agent.Tool("edit_file",
                    "Replace an exact, unique string in a file inside the root "
                    "(preferred over write_file for small changes). Mutating -- "
                    "requires confirmation.",
                    _EDIT_SCHEMA, paid(edit_file), paid=True),
        _agent.Tool("run",
                    "Run a shell command (/bin/sh -c) with the working directory "
                    "set to the project root; returns exit code + captured output. "
                    "Use for tests/build/git-mutations. Requires confirmation.",
                    _RUN_SCHEMA, run_invoke, paid=True),
        _agent.Tool("git",
                    "Run a read-only git subcommand (status/diff/log/show/...) in "
                    "the project root. For commits/adds use the run tool.",
                    _GIT_SCHEMA, git_invoke, paid=False),
    ]

    if include_search and client is not None and _index.discover_store(None) is not None:
        def search_invoke(arguments, *, confirm: bool = False):
            return _mcp.search_tool(client, **_clean(arguments))
        tools.append(_agent.Tool(
            "project_search",
            "Semantic search over the project's local .venice index (built by "
            "`venice index`) for code relevant to a natural-language query. "
            "Read-only.",
            _SEARCH_SCHEMA, search_invoke, paid=False,
        ))
    return tools


def tool_names(tools: List[_agent.Tool]) -> str:
    """Comma-joined tool names, for a startup banner (stderr)."""
    return ", ".join(t.name for t in tools)
