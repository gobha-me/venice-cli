"""Persistent agent memory + task store for `venice chat`/`venice code` (#49).

Two durable stores the agent maintains itself, so multi-step and cross-session work
-- and the #52 planner handing state to subagents -- survives beyond one transcript:

* **Memory** -- named notes the agent writes and recalls. TWO TIERS:
    - ``project`` (default): rides the repo at ``<root>/.venice/memory/memory.json``
      (root discovered by walking up from cwd for ``.venice/``, like the index), so
      subagents in the same tree share it.
    - ``global``: user-global at ``<MEMORY_DIR>/memory.json`` (``$VENICE_MEMORY_DIR``
      overrides), so knowledge that isn't project-specific travels with the agent.
* **Tasks** -- a lightweight project-only checklist (no scope: work items belong to a
  run/repo, not the user) at ``<project>/.venice/memory/tasks.json``.

Both files are name-keyed JSON docs written 0600 via an atomic tmp+fsync+rename
(mirrors ``_session.save`` / ``_index.save_store``; no shared helper exists -- each
store reimplements). Recall is **plain substring** over name+description+body -- zero
new deps, always works offline (semantic recall via the ``.venice`` index is a
deferred enhancement).

Concurrency: each mutation reads the doc, edits its in-memory copy, and atomically
rewrites it (read-modify-write). Within one process the four mutators hold a module
``_STORE_LOCK`` across that RMW, so #52 ``--parallel`` subagents sharing one project
store can't drop each other's change (added when parallel dispatch shipped; today the
role-scoped workers get no memory/task tools, so it's forward-looking insurance). Reads
stay lock-free -- the atomic ``os.replace`` in ``_save`` means a reader never sees a torn
file. The lock is **in-process only**: two separate ``venice`` processes writing the same
store remain last-writer-wins (like ``secrets.json`` / ``sessions`` / the index) -- guard
that with per-entry files or an OS lock if it ever becomes real; callers need not change.

Hygiene (CLAUDE.md): an entry NAME is validated like a persona/secret name (no
traversal) and refused if secret-shaped (``is_secret_path``) so the store can't be
used to label/stash credentials; the name is only ever a dict key, so there is no
filesystem-path surface. The store lives under ``.venice/`` -- already pruned by
``_index.walk_files`` and refused by the coding tools' ``is_protected_dir_path`` guard
-- so the agent cannot reach ``memory.json`` out-of-band.
"""
from __future__ import annotations

import json
import os
import re
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from .. import config
from . import _index

MEMORY_VERSION = 1
TASKS_VERSION = 1
TASK_STATUSES = ("pending", "in_progress", "done")
SCOPES = ("project", "global")

MAX_CONTENT_CHARS = 8192  # a single memory body (~2k tokens); keeps the store small
_MAX_NAME_LEN = 128
_MAX_TASK_TEXT_CHARS = 2048
_PREVIEW_CHARS = 200
_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")  # mirrors auth._NAME_RE

# #52 --parallel: serialize each read-modify-write mutation so concurrent in-process
# subagents can't clobber one another's change (see the module docstring). One module
# lock is enough -- mutation frequency is trivially low. In-process only.
_STORE_LOCK = threading.Lock()


class MemStoreError(Exception):
    """A memory name/scope/status is invalid, or the store is malformed. Printable.

    Named ``MemStoreError`` (not ``MemoryError``) to avoid shadowing the builtin.
    """


def _now_iso() -> str:
    """UTC ISO-8601 (microsecond), lexically sortable. Mirrors ``_session._now_iso``."""
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


# --------------------------------------------------------------------------- #
# Scope -> directory (resolved per call, never at import -- no side effects)
# --------------------------------------------------------------------------- #
def _global_dir() -> Path:
    """The user-global memory dir: ``$VENICE_MEMORY_DIR`` or ``config.MEMORY_DIR``."""
    return Path(os.environ.get(config.ENV_MEMORY_DIR) or config.MEMORY_DIR)


def _project_dir(start: Optional[Path] = None) -> Path:
    """The project memory dir: ``<root>/.venice/memory``.

    Walks up from `start` (default cwd) for an existing ``.venice/`` directory
    (git-style, mirrors ``_index.discover_store``), stopping at the filesystem root
    and never above ``$HOME``; on a miss, defaults to ``<cwd>/.venice/memory`` (created
    on write). `start` is exposed only for testability.
    """
    start = Path(start or os.getcwd()).resolve()
    home = config.HOME.resolve()
    cur = start
    while True:
        if (cur / config.INDEX_DIRNAME).is_dir():
            return cur / config.INDEX_DIRNAME / config.MEMORY_SUBDIR
        if cur == cur.parent or cur == home:
            break
        cur = cur.parent
    return start / config.INDEX_DIRNAME / config.MEMORY_SUBDIR


def _check_scope(scope: str) -> str:
    if scope not in SCOPES:
        raise MemStoreError(f"unknown scope {scope!r} (use 'project' or 'global')")
    return scope


def _scope_dir(scope: str, *, start: Optional[Path] = None) -> Path:
    if _check_scope(scope) == "global":
        return _global_dir()
    return _project_dir(start)


def _memory_file(scope: str, *, start: Optional[Path] = None) -> Path:
    return _scope_dir(scope, start=start) / config.MEMORY_FILENAME


def _tasks_file(*, start: Optional[Path] = None) -> Path:
    return _project_dir(start) / config.TASKS_FILENAME  # tasks are project-only


# --------------------------------------------------------------------------- #
# Store I/O (atomic single-file, 0600 -- mirrors _session.save / _index.save_store)
# --------------------------------------------------------------------------- #
def _fresh(kind: str) -> dict:
    if kind == "memory":
        return {"venice_memory": MEMORY_VERSION, "entries": {}}
    return {"venice_tasks": TASKS_VERSION, "tasks": []}


def _load(path: Path, *, kind: str, strict: bool) -> dict:
    """Read a store doc. Missing -> fresh. Malformed -> raise (strict, so a write
    never clobbers a corrupt store) or warn+fresh (tolerant, so a read still works)."""
    try:
        raw = Path(path).read_text(encoding="utf-8")
    except FileNotFoundError:
        return _fresh(kind)
    except OSError as e:
        if strict:
            raise MemStoreError(f"cannot read memory store {path}: {e}")
        print(f"venice: cannot read memory store {path}: {e}", file=sys.stderr)
        return _fresh(kind)
    try:
        doc = json.loads(raw)
    except ValueError as e:
        if strict:
            raise MemStoreError(f"malformed memory store {path}: {e}")
        print(f"venice: ignoring malformed memory store {path}: {e}", file=sys.stderr)
        return _fresh(kind)
    container = "entries" if kind == "memory" else "tasks"
    empty = {} if kind == "memory" else []
    if not isinstance(doc, dict) or not isinstance(doc.get(container), type(empty)):
        if strict:
            raise MemStoreError(f"malformed memory store {path} (expected an object)")
        print(f"venice: ignoring malformed memory store {path}", file=sys.stderr)
        return _fresh(kind)
    return doc


def _maybe_gitignore(store_dir: Path) -> None:
    """For the PROJECT tier (``<root>/.venice/memory``) ensure ``<root>/.venice/.gitignore``
    ignores the store -- local-by-default, matching the index. No-op for the user-global
    tier (whose parent is ``~/.config/venice``, not ``.venice``). Best-effort, never fatal."""
    venice_dir = Path(store_dir).parent
    if venice_dir.name != config.INDEX_DIRNAME:
        return
    try:
        gi = venice_dir / ".gitignore"
        if not gi.exists():
            gi.write_text(
                "# Venice agent memory + tasks -- machine-generated, do not commit.\n*\n",
                encoding="utf-8")
    except OSError:
        pass


def _save(path: Path, doc: dict) -> Path:
    """Atomically write `doc` to `path` (mode 0600): mkdir 0700 -> tmp 0600 -> dump/
    fsync -> atomic rename -> chmod 0600. Mirrors ``_session.save``."""
    path = Path(path)
    store_dir = path.parent
    store_dir.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(store_dir, 0o700)
    except OSError:
        pass
    _maybe_gitignore(store_dir)
    tmp = path.with_suffix(".tmp")
    fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=2, sort_keys=True)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    os.replace(tmp, path)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    return path


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def _valid_name(name) -> str:
    if not isinstance(name, str) or not name.strip():
        raise MemStoreError("memory name is required")
    name = name.strip()
    if name in (".", ".."):
        raise MemStoreError(f"invalid memory name {name!r}")
    if len(name) > _MAX_NAME_LEN:
        raise MemStoreError(f"memory name too long (max {_MAX_NAME_LEN} chars)")
    if not _NAME_RE.match(name):
        raise MemStoreError(
            f"invalid memory name {name!r} (allowed: letters, digits, '_', '.', '-')")
    if _index.is_secret_path(name):
        raise MemStoreError(f"refusing to store a secret-shaped memory name {name!r}")
    return name


# --------------------------------------------------------------------------- #
# Memory entries
# --------------------------------------------------------------------------- #
def _meta(name: str, scope: str, entry: dict) -> dict:
    """Metadata projection (no body) -- the always-loaded index for `list`/`search`."""
    return {
        "name": name,
        "scope": scope,
        "type": entry.get("type", "note"),
        "description": entry.get("description", ""),
        "created": entry.get("created"),
        "updated": entry.get("updated"),
    }


def _preview(content: str, query: str) -> str:
    """A one-line snippet around the match (or head of the body) for search results."""
    if not content:
        return ""
    lc = content.lower()
    idx = lc.find((query or "").strip().lower())
    if idx < 0:
        snippet = content[:_PREVIEW_CHARS]
    else:
        head = max(0, idx - 40)
        snippet = content[head:head + _PREVIEW_CHARS]
    snippet = " ".join(snippet.split())
    if len(snippet) < len(content):
        snippet += " …"
    return snippet


def write_entry(name, content, *, scope: str = "project", type=None,
                description=None, start: Optional[Path] = None) -> dict:
    """Create or overwrite a memory entry. Returns its metadata (no body)."""
    name = _valid_name(name)
    scope = _check_scope(scope)
    content = "" if content is None else str(content)
    if len(content) > MAX_CONTENT_CHARS:
        raise MemStoreError(
            f"memory content too long ({len(content)} chars; max {MAX_CONTENT_CHARS})")
    path = _memory_file(scope, start=start)
    with _STORE_LOCK:  # serialize the read-modify-write (#52 --parallel)
        doc = _load(path, kind="memory", strict=True)
        entries: Dict[str, dict] = doc["entries"]
        now = _now_iso()
        prev = entries.get(name)
        created = (
            prev["created"] if isinstance(prev, dict) and prev.get("created") else now
        )
        entries[name] = {
            "content": content,
            "type": (str(type).strip() if type else "note"),
            "description": (str(description).strip() if description else ""),
            "created": created,
            "updated": now,
        }
        _save(path, doc)
        meta = _meta(name, scope, entries[name])
    return meta


def read_entry(name, *, scope: Optional[str] = None,
               start: Optional[Path] = None) -> Optional[dict]:
    """Read one entry (name, scope, body + metadata). `scope=None` tries project then
    global, returning the first hit. None if not found."""
    name = _valid_name(name)
    scopes = [_check_scope(scope)] if scope else ["project", "global"]
    for sc in scopes:
        path = _memory_file(sc, start=start)
        entry = _load(path, kind="memory", strict=False)["entries"].get(name)
        if isinstance(entry, dict):
            return {**_meta(name, sc, entry), "content": entry.get("content", "")}
    return None


def list_entries(*, scope: Optional[str] = None,
                 start: Optional[Path] = None) -> List[dict]:
    """Metadata for every entry (no bodies). `scope=None` lists both tiers, tagged."""
    scopes = [_check_scope(scope)] if scope else ["project", "global"]
    out: List[dict] = []
    for sc in scopes:
        path = _memory_file(sc, start=start)
        for name, entry in _load(path, kind="memory", strict=False)["entries"].items():
            if isinstance(entry, dict):
                out.append(_meta(name, sc, entry))
    out.sort(key=lambda m: (m["name"], m["scope"]))
    return out


def search_entries(query, *, scope: Optional[str] = None,
                   start: Optional[Path] = None) -> List[dict]:
    """Plain substring search over name+description+body. `scope=None` searches both
    tiers. Each result is metadata + a `preview` snippet (no full body)."""
    q = (query or "").strip().lower()
    if not q:
        raise MemStoreError("search query is required")
    scopes = [_check_scope(scope)] if scope else ["project", "global"]
    out: List[dict] = []
    for sc in scopes:
        path = _memory_file(sc, start=start)
        for name, entry in _load(path, kind="memory", strict=False)["entries"].items():
            if not isinstance(entry, dict):
                continue
            content = entry.get("content", "") or ""
            hay = "\n".join([name, entry.get("description", "") or "", content]).lower()
            if q in hay:
                m = _meta(name, sc, entry)
                m["preview"] = _preview(content, q)
                out.append(m)
    out.sort(key=lambda m: (m["name"], m["scope"]))
    return out


def delete_entry(name, *, scope: str = "project",
                 start: Optional[Path] = None) -> bool:
    """Delete an entry. Returns True if it existed."""
    name = _valid_name(name)
    scope = _check_scope(scope)
    path = _memory_file(scope, start=start)
    with _STORE_LOCK:  # serialize the read-modify-write (#52 --parallel)
        doc = _load(path, kind="memory", strict=True)
        if name in doc["entries"]:
            del doc["entries"][name]
            _save(path, doc)
            return True
    return False


# --------------------------------------------------------------------------- #
# Tasks (project-only)
# --------------------------------------------------------------------------- #
def _check_status(status) -> str:
    if status not in TASK_STATUSES:
        raise MemStoreError(
            f"unknown status {status!r} (use one of: {', '.join(TASK_STATUSES)})")
    return status


def _check_text(text) -> str:
    text = (str(text).strip() if text is not None else "")
    if not text:
        raise MemStoreError("task text is required")
    if len(text) > _MAX_TASK_TEXT_CHARS:
        raise MemStoreError(f"task text too long (max {_MAX_TASK_TEXT_CHARS} chars)")
    return text


def add_task(text, *, start: Optional[Path] = None) -> dict:
    """Append a new `pending` task. Returns the created task."""
    text = _check_text(text)
    path = _tasks_file(start=start)
    with _STORE_LOCK:  # serialize the read-modify-write (#52 --parallel; also next_id)
        doc = _load(path, kind="tasks", strict=True)
        tasks: List[dict] = doc["tasks"]
        next_id = 1
        for t in tasks:
            try:
                next_id = max(next_id, int(t.get("id", 0)) + 1)
            except (TypeError, ValueError):
                pass
        now = _now_iso()
        task = {"id": str(next_id), "text": text, "status": "pending",
                "created": now, "updated": now}
        tasks.append(task)
        _save(path, doc)
    return dict(task)


def update_task(task_id, *, status=None, text=None,
                start: Optional[Path] = None) -> dict:
    """Update a task's status and/or text. Raises if the id is unknown."""
    if status is None and text is None:
        raise MemStoreError("nothing to update (pass status and/or text)")
    if status is not None:
        _check_status(status)
    tid = str(task_id).strip()
    if not tid:
        raise MemStoreError("task id is required")
    path = _tasks_file(start=start)
    with _STORE_LOCK:  # serialize the read-modify-write (#52 --parallel)
        doc = _load(path, kind="tasks", strict=True)
        for t in doc["tasks"]:
            if str(t.get("id")) == tid:
                if status is not None:
                    t["status"] = status
                if text is not None:
                    t["text"] = _check_text(text)
                t["updated"] = _now_iso()
                _save(path, doc)
                return dict(t)
    raise MemStoreError(f"no task with id {tid!r}")


def list_tasks(*, status=None, start: Optional[Path] = None) -> List[dict]:
    """All tasks (creation order), optionally filtered by status."""
    if status is not None:
        _check_status(status)
    path = _tasks_file(start=start)
    tasks = [dict(t) for t in _load(path, kind="tasks", strict=False)["tasks"]
             if isinstance(t, dict)]
    if status is not None:
        tasks = [t for t in tasks if t.get("status") == status]
    return tasks
