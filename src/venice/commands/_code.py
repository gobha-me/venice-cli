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
  exec's boundary is the confirm gate + cwd + timeout + env-scrub + the optional
  allow/deny **shell policy**, **not** path containment -- which is why it is always
  gated. ``git`` exposes only read-only subcommands freely; mutations go through the
  (gated) ``run`` tool. The exec rails (``run_cmd``/``git_cmd``/``_scrubbed_env``/the
  policy) live in :mod:`commands._exec` so `venice chat --shell` (#33) shares the
  exact same gate.
"""
from __future__ import annotations

import dataclasses
import fnmatch
import os
import re
import threading
from pathlib import Path
from typing import List, Optional

from . import _agent, _exec, _index, _mcp, _memory
from ._exec import (  # shared exec rails (#33): one gate for `code` and chat --shell
    DEFAULT_EXEC_TIMEOUT,
    MAX_OUTPUT_CHARS,
    _confirm,
    _err,
    _GIT_SCHEMA,
    _obj,
    _ok,
    _p,
    _RUN_SCHEMA,
    _scrubbed_env,
    _SECRET_ENV,
    git_cmd,
    run_cmd,
)

# --------------------------------------------------------------------------- #
# Limits + constants  (exec limits `MAX_OUTPUT_CHARS`/`DEFAULT_EXEC_TIMEOUT`,
# `_SECRET_ENV`, and the `_ok`/`_err`/`_confirm` helpers come from `_exec`)
# --------------------------------------------------------------------------- #
MAX_READ_BYTES = _index.MAX_FILE_BYTES        # reuse the indexer's oversize cap
MAX_GREP_MATCHES = 200
MAX_GREP_FILES = 5_000

# Loop-controlled kwargs the model must never supply (mirrors _agent._CONTROLLED).
_CONTROLLED = ("confirm", "max_spend", "output_dir")


def _clean(arguments) -> dict:
    if not isinstance(arguments, dict):
        return {}
    return {k: v for k, v in arguments.items() if k not in _CONTROLLED}


class _PathError(Exception):
    """A path failed the sandbox / denylist checks. Its message is printable."""


# --------------------------------------------------------------------------- #
# Attachable roots + path sandbox (#76)
# --------------------------------------------------------------------------- #
class Roots:
    """The directories `venice code`'s file tools may touch, plus the *active* root
    that relative paths and the `run`/`git` cwd resolve against (#76).

    - ``base`` is the active root ("cwd"): relative paths join here and the shell
      tools run here. :meth:`attach` moves it (with ``activate``).
    - ``allow`` are the readable **and** writable roots (``base`` is always one).
    - ``deny`` roots are excluded from **writes** (deny wins), so a deny root nested
      under an allow root is readable but not writable.

    A guardrail, **not** a sandbox: writes that land outside the writable set fail
    LOUDLY (naming the roots) so a well-behaved agent catches its own cross-repo
    mistake instead of leaking files silently -- the exact signal missing from the
    incident that motivated #76. All roots are ``realpath``-resolved on construction
    and mutation so symlinked roots compare correctly (the invariant
    :func:`_index.resolves_inside` assumes). This is a mutable, per-session holder:
    the tool closures capture one instance and :meth:`attach` grows it at runtime.
    """

    def __init__(self, base, allow=(), deny=()):
        self.base = os.path.realpath(base)
        self.allow: List[str] = []
        self.deny: List[str] = []
        self._add(self.allow, self.base)
        for r in allow:
            self._add(self.allow, r)
        for r in deny:
            self._add(self.deny, r)

    @classmethod
    def single(cls, root):
        """A one-root holder with no extra allow/deny -- the legacy single-root case
        (and the read-only scout's inner toolset), behaviourally identical to the
        pre-#76 sandbox."""
        return cls(root)

    @staticmethod
    def _add(bucket: List[str], path) -> str:
        real = os.path.realpath(path)
        if real not in bucket:
            bucket.append(real)
        return real

    def _containing(self, real):
        """The longest allow root that contains `real` (symlinks already resolved),
        or None when the path is outside every allow root."""
        hits = [r for r in self.allow if _index.resolves_inside(Path(real), Path(r))]
        return max(hits, key=len) if hits else None

    def check(self, real, *, write: bool) -> str:
        """Return the allow root containing `real`, or raise :class:`_PathError`.

        A read must land inside some allow root; a write must additionally land
        outside every deny root (deny wins). The message names the roots so the
        failure is a usable signal, not a silent redirect.
        """
        root = self._containing(real)
        if root is None:
            kind = "writable" if write else "readable"
            raise _PathError(
                f"path escapes the {kind} roots {self.allow}; attach it with "
                f"attach_root first: {real}"
            )
        if write:
            for d in self.deny:
                if _index.resolves_inside(Path(real), Path(d)):
                    raise _PathError(
                        f"path is under a read-only (deny) root {d}: {real}"
                    )
        return root

    def attach(self, path, *, write: bool = True, activate: bool = True) -> str:
        """Register `path` as an additional root (and, by default, make it active).

        `path` is resolved against the current ``base`` (or used as-is if absolute)
        and is NOT sandbox-confined -- this is the tool that *widens* the sandbox.
        A writable attach joins ``allow``; a read-only attach (``write=False``) joins
        both ``allow`` and ``deny`` (readable but not writable). Returns the realpath.
        Raises :class:`_PathError` for a non-directory or a reckless target
        (filesystem root / ``$HOME`` exactly).
        """
        a = str(path or "").strip()
        if not a:
            raise _PathError("path is required")
        real = os.path.realpath(a if os.path.isabs(a) else os.path.join(self.base, a))
        if not os.path.isdir(real):
            raise _PathError(f"not a directory: {path}")
        home = os.path.realpath(os.path.expanduser("~"))
        if real == os.path.realpath(os.sep) or real == home:
            raise _PathError(f"refusing to attach a top-level root: {real}")
        self._add(self.allow, real)
        if not write:
            self._add(self.deny, real)
        if activate:
            self.base = real
        return real


# --------------------------------------------------------------------------- #
# Path sandbox
# --------------------------------------------------------------------------- #
def _safe_path(roots: "Roots", arg, *, must_exist: bool = False, write: bool = False):
    """Resolve `arg` (relative to the active root) and enforce the roots + denylists.

    Returns ``(real_abspath, rel_posix)``. Raises :class:`_PathError` if the path
    escapes the allowed roots (or, for a write, lands in a deny root), is
    secret-shaped, lives under a protected dir, or (when `must_exist`) does not
    exist. `rel` is relative to the containing allow root. `roots` holds
    already-``realpath``-resolved roots.
    """
    if arg is None or not str(arg).strip():
        raise _PathError("path is required")
    a = str(arg)
    joined = a if os.path.isabs(a) else os.path.join(roots.base, a)
    real = os.path.realpath(os.path.normpath(joined))
    container = roots.check(real, write=write)  # raises if outside / deny
    rel = Path(os.path.relpath(real, container)).as_posix()
    if _index.is_secret_path(rel) or _index.is_protected_dir_path(rel):
        raise _PathError(f"path is in a protected location (secret/.git/.venice): {arg}")
    if must_exist and not os.path.exists(real):
        raise _PathError(f"no such file or directory: {arg}")
    return real, rel


def _stage_write(target: str, text: str) -> str:
    """Write `text` to a temp file next to `target` and return the temp path.

    Does the durable part of an atomic write (tmp in the same dir + fsync) but
    does **not** replace `target` -- the caller commits with ``os.replace`` once
    every file in a batch has staged (so a multi-file write is all-or-nothing;
    see `apply_patch`). Normal file mode (umask-respecting) -- these are source
    files, not the 0600 secrets `_index.save_store` writes. Creates parent dirs
    (already inside root). On failure the temp file is cleaned up before raising.
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
    return tmp


def _atomic_write(target: str, text: str) -> None:
    """Write `text` to `target` atomically (stage a tmp file, then replace)."""
    os.replace(_stage_write(target, text), target)


# --------------------------------------------------------------------------- #
# Tool implementations (print-free; return JSON-serializable dicts)
# --------------------------------------------------------------------------- #
def read_file(roots, path, *, offset=None, limit=None) -> dict:
    try:
        real, rel = _safe_path(roots, path, must_exist=True)
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


def list_dir(roots, *, path=".") -> dict:
    try:
        real, rel = _safe_path(roots, path or ".", must_exist=True)
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
    roots, pattern, *, path=None, glob=None, ignore_case=False,
    max_matches: int = MAX_GREP_MATCHES,
) -> dict:
    if not pattern or not str(pattern).strip():
        return _err("pattern is required")
    try:
        rx = re.compile(str(pattern), re.IGNORECASE if ignore_case else 0)
    except re.error as e:
        return _err(f"invalid regex: {e}")

    single_file = None
    walk_root = roots.base
    if path:
        try:
            real, _rel = _safe_path(roots, path, must_exist=True)
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
        rel = Path(os.path.relpath(str(fp), roots.base)).as_posix()
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


def write_file(roots, path, content, *, confirm: bool = False) -> dict:
    if content is None:
        return _err("content is required")
    try:
        real, rel = _safe_path(roots, path, write=True)
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


def edit_file(roots, path, old, new, *, confirm: bool = False) -> dict:
    if old is None or new is None:
        return _err("both old and new are required")
    if old == "":
        return _err("old must be a non-empty string")
    try:
        real, rel = _safe_path(roots, path, must_exist=True, write=True)
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


def _apply_one_hunk(text: str, hunk: dict, index: int):
    """Apply one {old,new[,occurrence]} hunk to `text`.

    Returns (new_text, error_message). `occurrence` (1-based) picks which match
    to replace when `old` is not unique; without it, a non-unique `old` is an
    error (same rule as edit_file). Errors name the hunk index so the model can
    retry surgically.
    """
    old = hunk.get("old")
    new = hunk.get("new")
    if old is None or new is None:
        return None, f"hunk {index}: both old and new are required"
    old, new = str(old), str(new)
    if old == "":
        return None, f"hunk {index}: old must be a non-empty string"
    occ = text.count(old)
    if occ == 0:
        return None, f"hunk {index}: old string not found"
    occurrence = hunk.get("occurrence")
    if occurrence is None:
        if occ > 1:
            return None, (
                f"hunk {index}: old string is not unique ({occ} occurrences); "
                "pass occurrence=N (1-based) or add surrounding context"
            )
        return text.replace(old, new, 1), None
    try:
        n = int(occurrence)
    except (TypeError, ValueError):
        return None, f"hunk {index}: occurrence must be an integer"
    if not 1 <= n <= occ:
        return None, f"hunk {index}: occurrence {n} out of range (1..{occ})"
    return _replace_nth(text, old, new, n), None


def _replace_nth(text: str, old: str, new: str, n: int) -> str:
    """Replace the n-th (1-based) occurrence of `old` with `new`."""
    start = -1
    for _ in range(n):
        start = text.find(old, start + 1)
    return text[:start] + new + text[start + len(old):]


def apply_patch(roots, patches, *, confirm: bool = False,
                dry_run: bool = False) -> dict:
    """Apply a batch of edits, grouped per file, atomically across all files.

    `patches` is a list of ``{path, edits: [{old, new, occurrence?}, ...]}``.
    Every file's hunks are validated and applied in order against the in-memory
    text first; only once every file validates are they written. The write phase
    stages each file's new text to a temp file and commits them all with
    ``os.replace`` only after every stage succeeded, so a failure part-way
    through never leaves some files written and others not (cross-file
    atomicity, #67). Hunks are checked sequentially -- a later hunk sees the
    result of the earlier ones.

    With ``dry_run=True`` the hunks are validated as usual but nothing is
    written and no confirmation is required: the return value previews the
    per-hunk old->new changes for every file.
    """
    if not isinstance(patches, list) or not patches:
        return _err("patches must be a non-empty list of {path, edits}")
    plan = []     # (real, rel, new_text, n_edits)
    preview = []  # {path, edits: [{index, old, new, occurrence?}, ...]}  (dry_run)
    for fi, entry in enumerate(patches):
        if not isinstance(entry, dict):
            return _err(f"patches[{fi}]: must be an object with path + edits")
        path = entry.get("path")
        edits = entry.get("edits")
        if not isinstance(edits, list) or not edits:
            return _err(f"patches[{fi}]: edits must be a non-empty list")
        try:
            real, rel = _safe_path(roots, path, must_exist=True, write=True)
        except _PathError as e:
            return _err(f"patches[{fi}]: {e}")
        if os.path.isdir(real):
            return _err(f"patches[{fi}]: {rel} is a directory")
        _data, text = _index.read_text(Path(real))
        if text is None:
            return _err(f"patches[{fi}]: {rel} is binary or unreadable")
        file_preview = []
        for hi, hunk in enumerate(edits):
            if not isinstance(hunk, dict):
                return _err(f"patches[{fi}] hunk {hi}: must be an object")
            text, err = _apply_one_hunk(text, hunk, hi)
            if err is not None:
                return _err(f"{rel}: {err}")
            change = {"index": hi, "old": str(hunk.get("old")),
                      "new": str(hunk.get("new"))}
            if hunk.get("occurrence") is not None:
                change["occurrence"] = hunk.get("occurrence")
            file_preview.append(change)
        plan.append((real, rel, text, len(edits)))
        preview.append({"path": rel, "edits": file_preview})
    if dry_run:
        return _ok(action="dry_run", files=preview,
                   total_edits=sum(len(p["edits"]) for p in preview))
    if not confirm:
        total = sum(n for _r, _l, _t, n in plan)
        files = ", ".join(rel for _r, rel, _t, _n in plan)
        return _confirm(
            f"apply_patch will apply {total} edit(s) across {len(plan)} file(s): {files}"
        )
    # Stage every file first; commit (os.replace) only once all staged, so a
    # failure part-way through leaves nothing written (cross-file atomicity).
    staged = []  # (tmp, real, rel, n_edits)
    for real, rel, text, n in plan:
        try:
            tmp = _stage_write(real, text)
        except OSError as e:
            for t, *_ in staged:  # nothing committed yet -- just drop the temps
                try:
                    os.unlink(t)
                except OSError:
                    pass
            return _err(f"write failed for {rel}: {e}")
        staged.append((tmp, real, rel, n))
    # Commit: same-dir renames, as atomic as the filesystem allows. A rename
    # failing here (after every file staged) is the one remaining narrow window.
    results = []
    for tmp, real, rel, n in staged:
        os.replace(tmp, real)
        results.append({"path": rel, "edits": n})
    return _ok(action="patched", files=results,
               total_edits=sum(r["edits"] for r in results))


def attach_root(roots: "Roots", path, *, write: bool = True,
                activate: bool = True) -> dict:
    """Register another project root so the file tools can work across repos (#76).

    Widens `roots` (and, by default, moves the active root into `path`) so a session
    that spans repos writes where it means to instead of silently leaking into the
    startup root. No filesystem side effect of its own -- the writes it enables still
    route through the confirm gate -- so it is free/unconfirmed; its effect is loudly
    reported (the result names the new writable + active roots).
    """
    try:
        real = roots.attach(path, write=bool(write) if write is not None else True,
                            activate=bool(activate) if activate is not None else True)
    except _PathError as e:
        return _err(str(e))
    return _ok(attached=real, writable=bool(write) if write is not None else True,
               base=roots.base, allow=list(roots.allow), deny=list(roots.deny))


# `_scrubbed_env`, `run_cmd`, `git_cmd`, and `_GIT_READONLY` now live in `_exec`
# (imported above) so the chat `--shell` tool shares the identical gate (#33).


# --------------------------------------------------------------------------- #
# JSON schemas (literals; confirm/max_spend/output_dir deliberately absent).
# `_p`/`_obj` and the exec schemas `_RUN_SCHEMA`/`_GIT_SCHEMA` come from `_exec`.
# --------------------------------------------------------------------------- #
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
_PATCH_HUNK = _obj(
    {
        "old": _p("string", "Exact text to replace."),
        "new": _p("string", "Replacement text."),
        "occurrence": _p(
            "integer",
            "Which match to replace (1-based) when `old` is not unique; omit to "
            "require a unique match.",
        ),
    },
    ["old", "new"],
)
_PATCH_SCHEMA = _obj(
    {
        "patches": {
            "type": "array",
            "description": "Edits grouped per file; applied atomically across all files.",
            "items": _obj(
                {
                    "path": _p("string", "File to edit, relative to the root."),
                    "edits": {
                        "type": "array",
                        "items": _PATCH_HUNK,
                        "description": "Hunks applied in order against the file.",
                    },
                },
                ["path", "edits"],
            ),
        },
        "dry_run": _p(
            "boolean",
            "Preview the per-hunk old->new changes for every file without "
            "writing anything (no confirmation needed). Use to self-check a "
            "patch before applying it.",
        ),
    },
    ["patches"],
)
# `_RUN_SCHEMA` / `_GIT_SCHEMA` are imported from `_exec` (shared with chat --shell).
_ATTACH_ROOT_SCHEMA = _obj(
    {
        "path": _p(
            "string",
            "Directory to attach as an additional project root (relative to the "
            "current active root, or absolute). Use when work spans repos.",
        ),
        "write": _p(
            "boolean",
            "Whether the attached root is writable (default true). Pass false to "
            "attach it read-only (readable, but writes there fail).",
        ),
        "activate": _p(
            "boolean",
            "Make the attached root the active root so relative paths and the run/git "
            "tools resolve there (default true). Pass false to keep the current cwd.",
        ),
    },
    ["path"],
)
_SEARCH_SCHEMA = _obj(
    {
        "query": _p("string", "Natural-language description of the code to find."),
        "k": _p("integer", "Number of results (default 8)."),
    },
    ["query"],
)
_REINDEX_SCHEMA = _obj({})  # no parameters -- rebuilds the discovered .venice index

# Scout subagent (#52 slice 1): defaults for the per-scout tool-call budget.
_SCOUT_MAX_TOOL_CALLS = 6
_SCOUT_HARD_CAP = 15
_SCOUT_SCHEMA = _obj(
    {
        "task": _p(
            "string",
            "The specific question or investigation to delegate. The scout starts "
            "from a FRESH context with only read-only tools and returns a structured "
            "report (findings / confidence / dead-ends / not-checked / verified-vs-"
            "hypothetical). Ask one focused thing.",
        ),
        "focus": _p(
            "string",
            "Optional hint: a file, directory, or subsystem to concentrate on "
            "(a hint, not a hard scope).",
        ),
        "task_id": _p(
            "string",
            "Optional id of the checklist task (from task_add) this dispatch serves; "
            "echoed on the report and in the venice_merge rollup so track and merge "
            "stay linked.",
        ),
        "max_tool_calls": {
            "type": "integer",
            "minimum": 1,
            "maximum": _SCOUT_HARD_CAP,
            "description": (
                f"Optional cap on the scout's tool calls (default "
                f"{_SCOUT_MAX_TOOL_CALLS}, hard max {_SCOUT_HARD_CAP}). Each call is "
                "a model turn -- keep it small."
            ),
        },
    },
    ["task"],
)

# Worker subagent (#52 slice 2): defaults for the per-worker tool-call budget (a worker
# does more than a scout's 6/15) + the role -> tool-category presets.
_SPAWN_MAX_TOOL_CALLS = 12
_SPAWN_HARD_CAP = 40
# Per-worker USD media spend cap (#52 spend slice): the default ceiling on the cumulative
# estimated cost of the paid media an `asset` worker generates before further paid calls
# are refused. Finite by default -- a `yes=True` worker can't stop to ask, so its media
# blast radius is bounded in dollars as well as in tool-calls and roots (#76). Raise or
# disable (<= 0) via --spawn-max-spend / defaults.code.spawn_max_spend. `code` workers
# grant no paid media, so this never bites them.
_SPAWN_MAX_SPEND = 2.00
_ROLE_CATEGORIES = {
    # 'code': read/write files, run commands, git, semantic search -- no paid media, so
    # this role is spend-free (writes/exec contained by Roots + shell policy).
    "code": {"fs", "exec", "vcs", "search"},
    # 'asset': generate/edit media (paid, only present with --assets) + inspect images
    # (vision), pick models (catalog), poll async jobs.
    "asset": {"image", "audio", "video", "catalog", "vision", "jobs"},
}
_SPAWN_SCHEMA = _obj(
    {
        "task": _p(
            "string",
            "The single, self-contained task to delegate to a fresh WORKER subagent "
            "(e.g. 'implement function X in path/y.py and add a unit test'). The worker "
            "starts from a FRESH context with a role-scoped subset of your tools and "
            "returns a structured report (outcome / changes / verified / follow-ups / "
            "blockers). Give it everything it needs -- it cannot see this conversation.",
        ),
        "role": {
            "type": "string",
            "enum": ["code", "asset"],
            "description": (
                "The worker's role, which selects its tools (default 'code'). 'code' = "
                "read/write files, run commands, git, search (no paid media). 'asset' = "
                "generate/edit images, audio, video (PAID -- only available if this "
                "session was started with --assets; its cumulative media spend is bounded "
                "by a per-worker USD cap, set by the operator via --spawn-max-spend)."
            ),
        },
        "categories": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Optional override of the role's tool categories (e.g. ['fs','vcs']). "
                "Always intersected with the tools this session actually has -- a worker "
                "can never exceed your own capabilities, and never gets subagent-"
                "spawning or root-widening tools."
            ),
        },
        "focus": _p(
            "string",
            "Optional hint: a file, directory, or subsystem to concentrate on "
            "(a hint, not a hard scope).",
        ),
        "task_id": _p(
            "string",
            "Optional id of the checklist task (from task_add) this dispatch "
            "implements; echoed on the report and in the venice_merge rollup so "
            "track and merge stay linked.",
        ),
        "max_tool_calls": {
            "type": "integer",
            "minimum": 1,
            "maximum": _SPAWN_HARD_CAP,
            "description": (
                f"Optional cap on the worker's tool calls (default "
                f"{_SPAWN_MAX_TOOL_CALLS}, hard max {_SPAWN_HARD_CAP})."
            ),
        },
    },
    ["task"],
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
    assets: bool = False,
    max_spend=None,
    config=None,
    shell_allow=(),
    shell_deny=(),
    allow_root=(),
    deny_root=(),
    browser: bool = False,
    browser_allow=(),
    browser_deny=(),
    memory: bool = False,
) -> List[_agent.Tool]:
    """Build the coding tools bound to a realpath-resolved project `root`.

    Read-only tools (`read_file`/`list_dir`/`grep`/`git`) are free; mutating tools
    (`write_file`/`edit_file`/`run`) are ``paid=True`` and route through the confirm
    gate. `project_search` (reusing the free `_mcp.search_tool`) is added only when
    `include_search` and a `client` are supplied and a `.venice` index exists.

    `allow_root`/`deny_root` (#76) seed the session's :class:`Roots`: `root` is the
    active (startup) root, `allow_root` adds extra readable+writable roots, and
    `deny_root` marks roots excluded from writes (deny wins). The agent can widen the
    set at runtime with the free `attach_root` tool, which also switches the active
    root so relative paths + `run`/`git` cwd follow it. Writes outside the writable
    set fail loudly (a guardrail, not a sandbox) so a cross-repo session can't leak
    files silently into the startup root.

    `shell_allow`/`shell_deny` (#33) apply the shared allow/deny policy to the `run`
    tool -- the same policy `venice chat --shell` enforces (see `_exec.check_policy`).
    Empty lists leave `run` unrestricted (only the confirm gate), preserving the prior
    behavior of autonomous `venice code` runs.

    `browser` (#71) appends the `web_fetch`/`browser_capture` rails, scoped by the
    `browser_allow`/`browser_deny` URL policy (see `_agent.browser_tools`).

    `memory` (#49) appends the persistent memory + task rails (`_agent.memory_tools`):
    free, local notes (two tiers) + a project task list the agent maintains across
    turns/sessions -- the shared state a #52 planner hands to subagents.

    When `assets` and a `client` are supplied, the in-process asset-generation tools
    (`venice_image`/`venice_image_edit`/`venice_sfx`/`venice_music`/`venice_tts`/
    `venice_upscale`/`venice_bg_remove`/`venice_video`) are folded in via `_agent.builtin_tools`,
    so the agent can create images/audio in the project. They are paid and route
    through the same confirm/spend gate; generated files land in
    ``$VENICE_MCP_OUTPUT_DIR`` or, by default, under `root`.
    """
    roots = Roots(root, allow=allow_root, deny=deny_root)
    root = roots.base  # startup-root snapshot for the asset/browser output dirs

    def free(fn):
        def invoke(arguments, *, confirm: bool = False):
            return fn(roots, **_clean(arguments))
        return invoke

    def paid(fn):
        def invoke(arguments, *, confirm: bool = False):
            return fn(roots, confirm=confirm, **_clean(arguments))
        return invoke

    def run_invoke(arguments, *, confirm: bool = False):
        # cwd follows the active root so the shell and file tools stay in sync (#76).
        return run_cmd(roots.base, confirm=confirm, exec_timeout=exec_timeout,
                       allow=shell_allow, deny=shell_deny, **_clean(arguments))

    def git_invoke(arguments, *, confirm: bool = False):
        return git_cmd(roots.base, exec_timeout=exec_timeout, **_clean(arguments))

    tools = [
        _agent.Tool("read_file",
                    "Read a UTF-8 text file inside the project root and return its "
                    "lines. Use before editing. Read-only.",
                    _READ_SCHEMA, free(read_file), paid=False,
                    category="fs", tags=("read",)),
        _agent.Tool("list_dir",
                    "List the entries of a directory inside the project root "
                    "(secret-shaped files are hidden). Read-only.",
                    _LIST_SCHEMA, free(list_dir), paid=False,
                    category="fs", tags=("read",)),
        _agent.Tool("grep",
                    "Search project files for a regular expression and return "
                    "matching path:line:text. Read-only.",
                    _GREP_SCHEMA, free(grep_files), paid=False,
                    category="fs", tags=("read", "search")),
        _agent.Tool("write_file",
                    "Create or overwrite a file inside the project root with new "
                    "contents. Mutating -- requires confirmation.",
                    _WRITE_SCHEMA, paid(write_file), paid=True,
                    category="fs", tags=("write",)),
        _agent.Tool("edit_file",
                    "Replace an exact, unique string in a file inside the root "
                    "(preferred over write_file for small changes). Mutating -- "
                    "requires confirmation.",
                    _EDIT_SCHEMA, paid(edit_file), paid=True,
                    category="fs", tags=("write",)),
        _agent.Tool("apply_patch",
                    "Apply a batch of edits grouped per file, atomically across "
                    "all files (validated first, then written all-or-nothing). "
                    "Prefer this over edit_file for multi-hunk changes or when a "
                    "string is not unique (use occurrence=N); use edit_file for a "
                    "single unique change. Pass dry_run=true to preview the "
                    "changes without writing. Mutating -- requires confirmation.",
                    _PATCH_SCHEMA, paid(apply_patch), paid=True,
                    category="fs", tags=("write",)),
        _agent.Tool("run",
                    "Run a shell command (/bin/sh -c) with the working directory "
                    "set to the project root; returns exit code + captured output. "
                    "Use for tests/build/git-mutations. Requires confirmation.",
                    _RUN_SCHEMA, run_invoke, paid=True,
                    category="exec", tags=("exec", "mutate")),
        _agent.Tool("git",
                    "Run a read-only git subcommand (status/diff/log/show/...) in "
                    "the project root. For commits/adds use the run tool.",
                    _GIT_SCHEMA, git_invoke, paid=False,
                    category="vcs", tags=("read",)),
        _agent.Tool("attach_root",
                    "Attach another directory as a project root when your work spans "
                    "repos, and (by default) make it the active root so relative "
                    "paths and run/git resolve there. Writes outside the writable "
                    "roots fail loudly -- attach the repo first instead of writing to "
                    "the wrong one. Read-only: use write=false to attach without "
                    "granting writes.",
                    _ATTACH_ROOT_SCHEMA, free(attach_root), paid=False,
                    category="fs", tags=("root",)),
    ]

    if include_search and client is not None and _index.discover_store(None) is not None:
        def search_invoke(arguments, *, confirm: bool = False):
            return _mcp.search_tool(client, **_clean(arguments))
        tools.append(_agent.Tool(
            "project_search",
            "Semantic search over the project's local .venice index (built by "
            "`venice index`) for code relevant to a natural-language query. "
            "Read-only. Results are a SNAPSHOT of the last index build -- call "
            "reindex after edits, or use grep for live matches.",
            _SEARCH_SCHEMA, search_invoke, paid=False,
            category="search", tags=("read",),
        ))

        def reindex_invoke(arguments, *, confirm: bool = False):
            return _mcp.reindex_tool(client, confirm=confirm)
        tools.append(_agent.Tool(
            "reindex",
            "Rebuild the project's .venice index so project_search reflects edits "
            "made this session (project_search is a snapshot; grep is live). "
            "Re-embeds only changed files. Takes no arguments. Paid -- requires "
            "confirmation.",
            _REINDEX_SCHEMA, reindex_invoke, paid=True,
            category="search", tags=("write",),
        ))

    # Free model-catalog lookups so the agent can pick a valid `model` for the
    # asset tools (and see cost/context limits) instead of guessing, plus
    # venice_vision so it can see images (screenshots/mockups/its own
    # generations) instead of working blind, plus the async job tools so it can
    # poll/fetch a background=true media render (need a client for the API calls).
    if client is not None:
        tools.extend(_agent.builtin_tools(
            client,
            # #50: the free catalog/vision/job tools, selected by capability
            # (== {venice_models, venice_model_details, venice_vision,
            # venice_job_status, venice_job_result}).
            only=_agent.select(categories={"catalog", "vision", "jobs"}),
            config=config))

    if assets and client is not None:
        asset_dir = os.environ.get("VENICE_MCP_OUTPUT_DIR") or root
        tools.extend(_agent.builtin_tools(
            client, max_spend=max_spend, output_dir=asset_dir,
            # #50: the image/audio/video generation tools, selected by capability
            # (== the 8 asset names incl. venice_image_edit/venice_video, which live
            # in _CODE_ASSET_BUILTINS -- select scans the union).
            only=_agent.select(categories={"image", "audio", "video"}),
            config=config,  # #58: asset tools honor defaults.<cmd>.*
        ))

    if browser:
        # web_fetch/browser_capture rails (#71): no Venice API, so no client needed.
        # Screenshots land in $VENICE_MCP_OUTPUT_DIR or under the project root.
        tools.extend(_agent.browser_tools(
            allow=browser_allow, deny=browser_deny,
            output_dir=os.environ.get("VENICE_MCP_OUTPUT_DIR") or root,
            config=config,
        ))

    if memory:
        # persistent memory + task rails (#49): free, local, no client needed. Project
        # notes/tasks ride root's .venice/ (cwd == root for `venice code`).
        tools.extend(_agent.memory_tools())
    return tools


# --------------------------------------------------------------------------- #
# Scout subagent (#52 slice 1): the read-only inner toolset + the `venice_scout`
# Tool that delegates a disposable investigation to `_agent.run_scout`.
# --------------------------------------------------------------------------- #
def read_only_tools(root: str, client=None, *, include_search: bool = False
                    ) -> List[_agent.Tool]:
    """The read-only subset of the coding toolset, bound to a realpath-resolved `root`:
    `read_file` / `list_dir` / `grep` / read-only `git`, plus `project_search` when a
    `.venice` index exists and a `client` is supplied.

    This is an *additive* builder, deliberately NOT a slice of :func:`code_tools`, so
    that factory (and its tests) stay untouched. It builds NO mutating/paid tool, NO
    scout tool, and NO `attach_root` -- the structural guarantee behind the scout
    subagent's read-only + no-self-spawn invariant (see :func:`_agent.run_scout`). The
    single, immovable :class:`Roots` (no writable extras, no attach) keeps the scout
    confined to one root exactly as before #76.
    """
    roots = Roots.single(root)

    def free(fn):
        def invoke(arguments, *, confirm: bool = False):
            return fn(roots, **_clean(arguments))
        return invoke

    def git_invoke(arguments, *, confirm: bool = False):
        return git_cmd(roots.base, exec_timeout=DEFAULT_EXEC_TIMEOUT, **_clean(arguments))

    tools = [
        _agent.Tool("read_file",
                    "Read a UTF-8 text file inside the project root and return its "
                    "lines. Read-only.",
                    _READ_SCHEMA, free(read_file), paid=False,
                    category="fs", tags=("read",)),
        _agent.Tool("list_dir",
                    "List the entries of a directory inside the project root "
                    "(secret-shaped files are hidden). Read-only.",
                    _LIST_SCHEMA, free(list_dir), paid=False,
                    category="fs", tags=("read",)),
        _agent.Tool("grep",
                    "Search project files for a regular expression and return "
                    "matching path:line:text. Read-only.",
                    _GREP_SCHEMA, free(grep_files), paid=False,
                    category="fs", tags=("read", "search")),
        _agent.Tool("git",
                    "Run a read-only git subcommand (status/diff/log/show/...) in "
                    "the project root.",
                    _GIT_SCHEMA, git_invoke, paid=False,
                    category="vcs", tags=("read",)),
    ]
    if include_search and client is not None and _index.discover_store(None) is not None:
        def search_invoke(arguments, *, confirm: bool = False):
            return _mcp.search_tool(client, **_clean(arguments))
        tools.append(_agent.Tool(
            "project_search",
            "Semantic search over the project's local .venice index for code "
            "relevant to a natural-language query. Read-only (a snapshot of the last "
            "index build; use grep for live matches).",
            _SEARCH_SCHEMA, search_invoke, paid=False,
            category="search", tags=("read",),
        ))
    return tools


def scout_tool(oai, model, root: str, client, base_kwargs, *,
               include_search: bool = True,
               web_tool: Optional[_agent.Tool] = None,
               default_max_tool_calls: int = _SCOUT_MAX_TOOL_CALLS,
               hard_cap: int = _SCOUT_HARD_CAP,
               dispatches: Optional[list] = None) -> _agent.Tool:
    """Build the `venice_scout` Tool: delegate a read-only investigation to a
    disposable subagent (context firewall, #52).

    The `invoke` closure clamps the tool-call budget (regardless of what the schema
    lets through), assembles a read-only inner toolset via :func:`read_only_tools`, and
    runs :func:`_agent.run_scout`, converting any error (including
    ``openai.OpenAIError`` from the nested loop) into a `{"status":"error"}` envelope.

    `paid=False` mirrors `venice_chat`: a bounded nested model call, not a media
    purchase. The confirm/spend gate exists to guard side effects, and a read-only
    scout has none; its cost is bounded by `max_tool_calls`.

    `web_tool`, when given (the #77 "docs scout": `venice code --web-search --scout`), is a
    pre-built `venice_web_search` Tool appended to the scout's read-only inner set so it can
    DISCOVER documentation as well as read the tree. It must be `paid=False` (it is), else
    `_agent.run_scout` refuses it.

    `dispatches`, when given (the `--planner` harness, #52), is the session's shared
    dispatch record list: every launched scout -- including one that errored -- is
    appended for the `venice_merge` rollup. `None` (the default) records nothing.
    """
    root = os.path.realpath(root)

    def invoke(arguments, *, confirm: bool = False):
        args = _clean(arguments)
        task = (args.get("task") or "").strip()
        if not task:
            return _err("scout requires a non-empty 'task'")
        focus = args.get("focus") or None
        tid = str(args["task_id"]).strip() if args.get("task_id") else None
        req = args.get("max_tool_calls")
        n = default_max_tool_calls if not isinstance(req, int) or req <= 0 else req
        n = max(1, min(int(n), hard_cap))
        inner = read_only_tools(root, client, include_search=include_search)
        if web_tool is not None:
            inner.append(web_tool)  # #77 "docs scout": read-only + web discovery
        try:
            out = _agent.run_scout(
                oai, model, task, inner, base_kwargs,
                max_tool_calls=n, focus=focus,
            )
        except Exception as e:  # incl. openai.OpenAIError from the nested loop
            out = _err(f"scout failed: {e}")
        if tid:
            out["task_id"] = tid  # echo: link the report to its checklist task
        if dispatches is not None:
            _record_dispatch(dispatches, "scout", task=task, role=None,
                             task_id=tid, out=out)
        return out

    return _agent.Tool(
        _agent.SCOUT_TOOL_NAME,
        "Delegate a read-only investigation to a disposable SCOUT subagent with a "
        "FRESH context and only read tools (read_file/list_dir/grep/git/search). It "
        "returns a structured report (findings / confidence / dead-ends / not-checked "
        "/ verified-vs-hypothetical) so heavy exploration does not pollute your "
        "context. Use it to answer where/how/what questions before you edit. The "
        "scout cannot edit files or run commands.",
        _SCOUT_SCHEMA, invoke, paid=False,
        category="agent", tags=("read", "spawn"),
    )


def web_search_tool(oai, model: str, *, models=None, search_model=None,
                    mode: str = "on") -> _agent.Tool:
    """Build the `venice_web_search` rail Tool (#77): DISCOVER docs on the web.

    Makes one Venice web-search completion (via `_agent.run_web_search`) against a
    `supportsWebSearch` model and returns the answer plus the cited URLs. The agent
    follows a citation with `web_fetch` (the `--browser` rail, #71), which keeps every
    fetched URL under the operator's `browser.*` policy -- search discovers, the browser
    policy governs what gets read.

    `paid=False` mirrors `venice_chat` (a bounded, billed sub-completion, not a media
    purchase): it MUST be free so a read-only SCOUT can carry it (`_agent.run_scout`
    rejects any paid tool). Cost is surfaced as best-effort `cost_estimate_usd` provenance
    and bounded by the caller's tool-call budget. Category `web` (shared with the browser
    rails) keeps it out of `_REGISTRY` AND out of every spawn WORKER's grant -- neither the
    `code` nor `asset` role category set includes `web`, so the blast-radius filter drops
    it structurally (the injection concern from #52: a worker following instructions
    injected via search results is the nightmare case).

    Model choice is operator-controlled (`search_model` / the coding `model`), resolved
    once here and never model-facing, so the model can't escalate to a costlier model.
    """
    resolved = _agent.resolve_web_search_model(models, search_model, model)

    def invoke(arguments, *, confirm: bool = False):
        query = (_clean(arguments).get("query") or "").strip()
        if not query:
            return _err("web_search requires a non-empty 'query'")
        if not resolved:
            return _err(
                "no web-search-capable model available; pass --web-search-model (or "
                "defaults.code.web_search_model) naming a model that advertises "
                "supportsWebSearch"
            )
        try:
            return _agent.run_web_search(oai, resolved, query, mode=mode, models=models)
        except Exception as e:  # incl. openai.OpenAIError from the completion
            return _err(f"web_search failed: {e}")

    return _agent.Tool(
        _agent.WEB_SEARCH_TOOL_NAME,
        "Search the web for documentation or answers and get back a short summary plus "
        "the source URLs it cited. Use it to DISCOVER pages you do not already have a URL "
        "for (API docs, library usage, an error message). To read a cited page in full, "
        "follow up with web_fetch (needs --browser). Read-only; cannot edit files or run "
        "commands.",
        _agent._WEB_SEARCH_SCHEMA, invoke, paid=False,
        category="web", tags=("read", "network"),
    )


def _meter(tool: _agent.Tool, cap: float, spent: list) -> _agent.Tool:
    """Wrap a paid `tool` in a per-worker spend meter over the shared `spent` accumulator.

    Refuses (`status="blocked"`) once `spent[0]` has reached `cap`; otherwise runs the
    tool and tallies the `cost_estimate_usd` its result reports (media tools; write/exec
    tools report none, so they never move the meter). `confirm` MUST be forwarded -- the
    worker runs `yes=True`, so a paid media tool needs `confirm=True` to actually purchase
    (else its own `check_spend` returns an unresolved `confirmation_required`).
    """
    inner = tool.invoke

    def metered(arguments, *, confirm: bool = False):
        if spent[0] >= cap:
            return {
                "status": "blocked",
                "message": (
                    f"spawn: per-worker media spend cap ${cap:.2f} reached "
                    f"(spent ${spent[0]:.4f}); stop spending and wrap up your report."
                ),
            }
        result = inner(arguments, confirm=confirm)
        if isinstance(result, dict):
            c = result.get("cost_estimate_usd")
            if isinstance(c, (int, float)) and not isinstance(c, bool) and c > 0:
                spent[0] += float(c)
        return result

    return dataclasses.replace(tool, invoke=metered)


def spawn_tool(oai, model, base_kwargs, parent_tools, *,
               default_max_tool_calls: int = _SPAWN_MAX_TOOL_CALLS,
               hard_cap: int = _SPAWN_HARD_CAP,
               max_spend: Optional[float] = None,
               dispatches: Optional[list] = None) -> _agent.Tool:
    """Build the `venice_spawn` Tool: delegate a bounded task to a disposable, write/
    paid-capable WORKER subagent (#52 slice 2).

    Where `venice_scout` is a read-only context firewall, this is the same firewall for
    *doers*. The worker draws a role-scoped subset of the PARENT's already-built
    `parent_tools`, so its writes flow through the same :class:`Roots` (allow-minus-deny,
    fail loud outside it, #76) and the same `run` shell policy -- it can never exceed the
    capabilities the operator granted this session. The `invoke` closure resolves the
    role (or an explicit `categories` override) to a category set, filters `parent_tools`
    to it -- excluding the `agent` category so no scout/spawn leaks in (no nested
    subagents), and the `root`-tagged `attach_root` so a worker can't widen its own roots
    -- clamps the tool-call budget, and runs :func:`_agent.run_spawn`.

    `paid=False` mirrors `venice_scout`/`venice_chat`: the tool itself makes a bounded
    nested model call, not a media purchase. Any paid *media* the worker generates spends
    through the parent's own paid tools -- but under the worker's `yes=True` loop the
    per-call confirm/`max_spend` gate is auto-approved, so a cumulative ceiling is enforced
    here instead (#52 spend slice).

    Per-worker USD media cap: `max_spend` (default `_SPAWN_MAX_SPEND`, `<= 0` disables)
    caps the cumulative *estimated* media spend. Each granted paid tool is wrapped in a
    spend meter sharing one accumulator: it refuses (`status="blocked"`) once the cap is
    reached and otherwise tallies the `cost_estimate_usd` its result reports. Cost is only
    known post-call, so the call that crosses the cap completes and the *next* is blocked
    (bounds further spend, doesn't preempt -- like `run_loop`'s own gate). The no-double-
    count relies on the worker running `yes=True` (so `_resolve_spend` never re-invokes a
    wrapped tool). `code`-role paid tools report no cost, so the cap never bites them --
    a code worker behaves exactly as before this slice. The final report carries
    `spent_usd`/`spend_cap_usd` for handoff provenance.

    `dispatches`, when given (the `--planner` harness, #52), is the session's shared
    dispatch record list: every launched worker -- including one that errored, which
    still owes its provenance -- is appended for the `venice_merge` rollup. `None`
    (the default) records nothing.
    """
    def invoke(arguments, *, confirm: bool = False):
        args = _clean(arguments)
        task = (args.get("task") or "").strip()
        if not task:
            return _err("spawn requires a non-empty 'task'")
        tid = str(args["task_id"]).strip() if args.get("task_id") else None
        role = (args.get("role") or "code").strip().lower()
        override = args.get("categories")
        if isinstance(override, list) and override:
            requested = {str(c).strip().lower() for c in override}
        else:
            requested = set(_ROLE_CATEGORIES.get(role, _ROLE_CATEGORIES["code"]))
        focus = args.get("focus") or None
        req = args.get("max_tool_calls")
        n = default_max_tool_calls if not isinstance(req, int) or req <= 0 else req
        n = max(1, min(int(n), hard_cap))

        granted = [
            t for t in parent_tools
            if t.category in requested
            and t.category != "agent"       # never scout/spawn -- no nested subagents
            and t.category != "web"          # #77: never web_search/browser -- a worker
                                             # following injected instructions from a page
                                             # is the nightmare case (deny-by-default)
            and "root" not in t.tags         # never attach_root -- can't widen roots
        ]
        if not granted:
            available = sorted({
                t.category for t in parent_tools
                if t.category and t.category not in ("agent", "web")
                and "root" not in t.tags
            })
            return _err(
                f"spawn: no tools available for role '{role}' (requested categories "
                f"{sorted(requested)}); this session offers: {available or ['(none)']}. "
                "Start venice code with --assets for media generation."
            )

        # Per-worker USD media cap: wrap each paid tool in a spend meter sharing one
        # accumulator. `<= 0` disables (identity pass-through == pre-slice behavior).
        raw = _SPAWN_MAX_SPEND if max_spend is None else max_spend
        cap = None if raw <= 0 else float(raw)
        spent = [0.0]
        if cap is not None:
            granted = [_meter(t, cap, spent) if t.paid else t for t in granted]

        try:
            out = _agent.run_spawn(
                oai, model, task, granted, base_kwargs,
                max_tool_calls=n, focus=focus, role=role,
            )
        except Exception as e:  # incl. openai.OpenAIError from the nested loop
            out = _err(f"spawn failed: {e}")
        # Report spend even on the error path -- a worker that spent then crashed still
        # owes the parent its provenance.
        if cap is not None and isinstance(out, dict):
            out["spent_usd"] = round(spent[0], 4)      # handoff provenance
            out["spend_cap_usd"] = cap
        if tid:
            out["task_id"] = tid  # echo: link the report to its checklist task
        if dispatches is not None:
            _record_dispatch(dispatches, "spawn", task=task, role=role,
                             task_id=tid, out=out)
        return out

    return _agent.Tool(
        _agent.SPAWN_TOOL_NAME,
        "Delegate a bounded task to a disposable WORKER subagent with a FRESH context "
        "and a role-scoped subset of your tools. Unlike venice_scout it CAN edit files "
        "and run commands (role 'code') or generate media (role 'asset', if --assets). "
        "It returns a structured report (outcome / changes / verified / follow-ups / "
        "blockers) so the implementation churn does not pollute your context. Its writes "
        "are confined to your writable roots (fail loud outside them) and it cannot spawn "
        "further subagents or widen roots; an 'asset' worker's cumulative media spend is "
        "capped in USD. Use it to hand off a self-contained unit of work you can verify "
        "from the report.",
        _SPAWN_SCHEMA, invoke, paid=False,
        category="agent", tags=("write", "spawn"),
    )


# Planner harness (#52 planner slice): the dispatch record list + the merge rollup.
#
# `venice code --planner` gives the parent session one shared, append-only list of
# dispatch records. `scout_tool`/`spawn_tool` append to it as dispatches return, so
# by merge time the harness already holds every report's parsed `fields` map (the
# v0.63 provenance) without the model re-typing anything. `merge_summary` is the
# deterministic heart: pure over (records, #49 task store) -> rollup + structural
# warnings; `merge_tool` exposes it to the planner as `venice_merge`, and the
# `--json` envelope embeds the same rollup so callers get it even if the model
# skipped the merge call. Category `agent` keeps it out of every worker's grant
# (spawn's own filter) -- merging is the planner's job, a worker can't do it.
_MERGE_SCHEMA = _obj({})  # no parameters -- the harness already holds the records
_DISPATCH_TASK_CHARS = 200  # task text kept per record (a label, not the transcript)

# #52 --parallel: scout/spawn dispatches complete on worker threads, so the seq compute-
# and-append below is a read-modify-write that must be atomic (list.append is GIL-atomic
# but `len(dispatches) + 1` then append is two steps -> duplicate/gapped seq under a race).
# Serialize the whole record so seq stays unique and gap-free; `merge_summary` is otherwise
# order-tolerant. In-process only (subagents are threads in one process).
_DISPATCH_LOCK = threading.Lock()


def _record_dispatch(dispatches: list, kind: str, *, task: str,
                     role: Optional[str], task_id: Optional[str], out: dict) -> None:
    """Append one dispatch record for the `venice_merge` rollup (#52 planner slice).

    Called by `scout_tool`/`spawn_tool` after every *launched* dispatch -- ok or
    error envelope alike (a failed dispatch still owes provenance); validation
    refusals (empty task, empty grant) never launched anything and are not recorded.
    Thread-safe: the seq compute-and-append is serialized (`--parallel` workers append
    concurrently).
    """
    task = task if len(task) <= _DISPATCH_TASK_CHARS else (
        task[:_DISPATCH_TASK_CHARS] + "...")
    with _DISPATCH_LOCK:
        dispatches.append({
            "seq": len(dispatches) + 1,
            "kind": kind,                              # 'scout' | 'spawn'
            "role": role,                              # spawn only; None for a scout
            "task_id": task_id,                        # checklist link; None if unlinked
            "task": task,
            "status": out.get("status"),
            "fields": out.get("fields"),               # the v0.63 parsed section map
            "tool_calls": out.get("tool_calls"),
            "truncated": out.get("truncated"),
            "spent_usd": out.get("spent_usd"),
        })


def merge_summary(dispatches: List[dict]) -> dict:
    """The consolidated merge rollup: dispatch records + task store + warnings.

    Pure and deterministic over its inputs -- no model call. Warnings are strictly
    *structural* facts (a task not done, a dispatch that errored/truncated, a
    task_id that matches no task); judging report *content* stays the planner's
    job. The #49 task store is read tolerantly: unreadable -> empty + a warning,
    never a raise (merge must always produce a rollup).
    """
    warnings: List[str] = []
    try:
        tasks = _memory.list_tasks()
    except Exception as e:
        tasks = []
        warnings.append(f"task store unreadable: {e}")
    if not dispatches:
        warnings.append("no dispatches recorded (nothing was delegated this session)")

    spent = 0.0
    errors = 0
    known = {str(t.get("id")) for t in tasks}
    dispatched = {str(d.get("task_id")) for d in dispatches if d.get("task_id")}
    for d in dispatches:
        tag = f"dispatch #{d.get('seq')} ({d.get('kind')})"
        if d.get("status") != "ok":
            errors += 1
            warnings.append(f"{tag} ended status={d.get('status')!r}")
        if d.get("truncated"):
            warnings.append(f"{tag} hit its tool-call cap; its report may be incomplete")
        tid = d.get("task_id")
        if tid is not None and str(tid) not in known:
            warnings.append(f"{tag} references unknown task_id {str(tid)!r}")
        c = d.get("spent_usd")
        if isinstance(c, (int, float)) and not isinstance(c, bool):
            spent += float(c)
    for t in tasks:
        status = t.get("status")
        if status == "done":
            continue  # a done task never dispatched was handled inline -- fine
        suffix = "" if str(t.get("id")) in dispatched else " and was never dispatched"
        text = str(t.get("text") or "")[:80]
        warnings.append(f"task {t.get('id')} still {status}{suffix}: {text}")

    return {
        "dispatches": dispatches,
        "totals": {
            "dispatches": len(dispatches),
            "spawns": sum(1 for d in dispatches if d.get("kind") == "spawn"),
            "scouts": sum(1 for d in dispatches if d.get("kind") == "scout"),
            "errors": errors,
            "spent_usd": round(spent, 4),
        },
        "tasks": tasks,
        "warnings": warnings,
    }


def merge_tool(dispatches: list) -> _agent.Tool:
    """Build the `venice_merge` Tool over the session's shared dispatch list."""
    def invoke(arguments, *, confirm: bool = False):
        return {"status": "ok", **merge_summary(dispatches)}

    return _agent.Tool(
        _agent.MERGE_TOOL_NAME,
        "Consolidated rollup of every scout/spawn dispatch this session: per-dispatch "
        "provenance (parsed report fields, task_id link, tool calls, spend), the "
        "current task checklist, totals, and structural warnings (tasks not done, "
        "dispatches that errored or were truncated, unknown task_ids). Call it after "
        "the last unit, resolve its warnings, and base your MERGE SUMMARY on it. "
        "Free and read-only -- it reports what already happened.",
        _MERGE_SCHEMA, invoke, paid=False,
        category="agent", tags=("read",),
    )


def tool_names(tools: List[_agent.Tool]) -> str:
    """Comma-joined tool names, for a startup banner (stderr)."""
    return ", ".join(t.name for t in tools)
