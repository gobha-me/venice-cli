"""Session store for `venice chat` / `venice code` (#47).

A *session* is one REPL conversation persisted as a single JSON **envelope** so a
resume restores its *settings* -- model, system prompt, generation kwargs /
``venice_parameters``, ``max_tool_calls``, the ``code`` sandbox root, and the
running usage/cost ledger -- not just its messages. This is the persistence
substrate the compaction (#48/#74) and multi-agent (#52) work build on, and it
discharges #75's deferred "usage survives ``--resume``" criterion.

Layout: one ``<SESSIONS_DIR>/<id>.json`` per session, auto-saved after every
committed turn via an atomic 0600 write (mirrors ``auth._save_secrets``). The
store lives under ``~/.config/venice/sessions/`` by convention; ``$VENICE_SESSIONS_DIR``
overrides it (resolved per call so this module has no import-time side effects).

Hygiene (CLAUDE.md): the envelope holds only messages + settings + usage -- never
the API key (which lives on the client, not in ``gen_kwargs``). Bare session ids
are validated like personas (no traversal) and re-checked with
:func:`_index.resolves_inside` so ``--resume ../credentials`` can't escape the zone.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

from .. import config
from . import _index

SESSION_VERSION = 1


class SessionError(Exception):
    """A session id/file is unsafe, missing, or malformed. Message is printable."""


def _sessions_dir() -> Path:
    """The session store dir: ``$VENICE_SESSIONS_DIR`` or ``config.SESSIONS_DIR``.

    Resolved per call (never at import) so the module stays side-effect-free and
    the env override is honored even when set after import (e.g. in tests).
    """
    return Path(os.environ.get(config.ENV_SESSIONS_DIR) or config.SESSIONS_DIR)


def _now_iso() -> str:
    """UTC ISO-8601 (microsecond) -- lexically sortable for `most_recent`/`ls`.

    Microsecond precision so two sessions touched in the same second still order by
    true recency (a second-granularity stamp would tie and make `--continue` pick
    an arbitrary one of them).
    """
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def new_id() -> str:
    """A fresh session id: ``YYYYmmddTHHMMSS-<6hex>`` (UTC).

    The timestamp prefix makes lexical order == chronological (so ids sort by age);
    the random suffix avoids a collision when two sessions start in the same second.
    """
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    return f"{stamp}-{os.urandom(3).hex()}"


def command_from_label(label: str) -> str:
    """Map a REPL label ("venice chat"/"venice code") to a command key."""
    return "code" if "code" in (label or "") else "chat"


@dataclass
class Session:
    """One persisted REPL session (the in-memory view of an envelope)."""

    id: str
    command: str
    created: str
    updated: str
    model: Optional[str] = None
    system: Optional[str] = None
    gen_kwargs: dict = field(default_factory=dict)
    root: Optional[str] = None
    label: str = "venice chat"
    max_tool_calls: Optional[int] = None
    usage: Optional[dict] = None
    messages: list = field(default_factory=list)

    def to_envelope(self) -> dict:
        return {
            "venice_session": SESSION_VERSION,
            "id": self.id,
            "created": self.created,
            "updated": self.updated,
            "command": self.command,
            "model": self.model,
            "system": self.system,
            "gen_kwargs": self.gen_kwargs,
            "root": self.root,
            "label": self.label,
            "max_tool_calls": self.max_tool_calls,
            "usage": self.usage,
            "messages": self.messages,
        }

    @classmethod
    def from_envelope(cls, d: dict) -> "Session":
        """Hydrate from an envelope dict, tolerant of missing/foreign keys."""
        now = _now_iso()
        gk = d.get("gen_kwargs")
        return cls(
            id=str(d.get("id") or new_id()),
            command=d.get("command") or "chat",
            created=d.get("created") or now,
            updated=d.get("updated") or now,
            model=d.get("model"),
            system=d.get("system"),
            gen_kwargs=gk if isinstance(gk, dict) else {},
            root=d.get("root"),
            label=d.get("label") or "venice chat",
            max_tool_calls=d.get("max_tool_calls"),
            usage=d.get("usage") if isinstance(d.get("usage"), dict) else None,
            messages=_validate_messages(d.get("messages") or []),
        )


def new_session(command: str, *, label: str = "venice chat", model=None,
                system=None, gen_kwargs=None, root=None, max_tool_calls=None,
                messages=None) -> Session:
    """Mint a brand-new active session (fresh id + timestamps)."""
    now = _now_iso()
    return Session(
        id=new_id(), command=command, created=now, updated=now,
        model=model, system=system,
        gen_kwargs=dict(gen_kwargs or {}), root=root, label=label,
        max_tool_calls=max_tool_calls, messages=list(messages or []),
    )


def _reject_unsafe(name: str) -> None:
    """Raise unless `name` is a safe bare session id (no path traversal)."""
    if not name:
        raise SessionError("no session id given")
    if "/" in name or "\\" in name or ".." in name or name != os.path.basename(name):
        raise SessionError(f"invalid session id {name!r} (bare ids only)")


def _validate_messages(data) -> list:
    """A transcript must be a JSON list of message objects (each with a role)."""
    if not isinstance(data, list) or not all(
        isinstance(m, dict) and "role" in m for m in data
    ):
        raise SessionError("transcript must be a JSON list of message objects")
    return data


def _read_json(path: str):
    try:
        raw = Path(path).read_text(encoding="utf-8")
    except OSError as e:
        raise SessionError(f"cannot read session {path}: {e}")
    try:
        return json.loads(raw)
    except ValueError as e:
        raise SessionError(f"invalid session JSON in {path}: {e}")


def _resolve_zone_path(session_id: str) -> Path:
    """Resolve a bare id to ``<zone>/<id>.json``, rejecting traversal/escape."""
    _reject_unsafe(session_id)
    zone = _sessions_dir()
    root = Path(os.path.realpath(zone))
    path = zone / (session_id + ".json")
    real = Path(os.path.realpath(path))
    if not _index.resolves_inside(real, root):
        raise SessionError(f"session id {session_id!r} escapes {zone}")
    return path


def load(ref: str, command: str) -> Session:
    """Load a session by id (in-place) or import a transcript/envelope file.

    Resolution order:

    1. `ref` is an existing **file path** -> import it, minting a NEW id (the
       original file is never written back): a bare JSON list is a legacy
       transcript (back-compat with ``--resume FILE``); a dict is an envelope.
    2. `ref` names a session in the store (``<zone>/<ref>.json``) -> hydrate it
       **in place** (keep the id so auto-save updates the same file).
    3. otherwise -> :class:`SessionError`.

    `command` is the caller's command; it fills the ``command`` field only when the
    source doesn't carry one (a bare legacy transcript).
    """
    if os.path.isfile(ref):
        data = _read_json(ref)
        if isinstance(data, list):
            return new_session(command, messages=_validate_messages(data))
        if isinstance(data, dict) and ("venice_session" in data or "messages" in data):
            s = Session.from_envelope(data)
            # An imported file becomes a new session -- don't clobber a store id.
            now = _now_iso()
            s.id, s.created, s.updated = new_id(), now, now
            return s
        # A file that is neither a list nor an envelope-shaped dict is a malformed
        # transcript (back-compat: `--resume FILE` always expected a message list).
        raise SessionError("transcript must be a JSON list of message objects")

    # Not an existing file: a value that is clearly a path (separator / .json / abs)
    # was meant as a transcript, so report it as one rather than as a bad id.
    if (os.sep in ref or (os.altsep and os.altsep in ref)
            or ref.endswith(".json") or os.path.isabs(ref)):
        raise SessionError(f"cannot read transcript {ref}: no such file")

    path = _resolve_zone_path(ref)
    if not path.is_file():
        raise SessionError(f"no session {ref!r} in {_sessions_dir()}")
    data = _read_json(str(path))
    if not isinstance(data, dict):
        raise SessionError(f"session {ref!r} is malformed (expected an envelope)")
    s = Session.from_envelope(data)
    s.id = ref  # in-place resume keeps the store identity
    return s


def save(session: Session) -> Path:
    """Atomically write `session` to ``<zone>/<id>.json`` (mode 0600).

    Mirrors ``auth._save_secrets``: mkdir 0700 -> create tmp 0600 -> dump/fsync ->
    atomic rename -> chmod 0600. Stamps ``updated``. Raises OSError on disk failure
    (the REPL auto-save swallows it with a warning rather than crashing).
    """
    session.updated = _now_iso()
    zone = _sessions_dir()
    zone.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(zone, 0o700)
    except OSError:
        pass
    path = zone / (session.id + ".json")
    tmp = path.with_suffix(".tmp")
    fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(session.to_envelope(), f, indent=2, default=str)
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


def _iter_sessions():
    """Yield a :class:`Session` per readable envelope in the zone (id = stem)."""
    try:
        entries = sorted(_sessions_dir().glob("*.json"))
    except OSError:
        return
    for entry in entries:
        try:
            data = json.loads(entry.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(data, dict):
            continue
        try:
            s = Session.from_envelope(data)
        except SessionError:
            continue
        s.id = entry.stem
        yield s


def list_sessions() -> List[Tuple[str, str, str, int, Optional[str]]]:
    """``(id, command, updated, msg_count, model)`` per session, newest first."""
    rows = [
        (s.id, s.command, s.updated, len(s.messages), s.model)
        for s in _iter_sessions()
    ]
    rows.sort(key=lambda r: r[2], reverse=True)
    return rows


def most_recent(command: str) -> Optional[Session]:
    """The newest session for `command` (for ``--continue``), or None."""
    best = None
    for s in _iter_sessions():
        if s.command != command:
            continue
        if best is None or s.updated > best.updated:
            best = s
    return best


def delete(session_id: str) -> bool:
    """Remove ``<zone>/<id>.json``; True if it existed, False if not."""
    path = _resolve_zone_path(session_id)
    try:
        path.unlink()
        return True
    except FileNotFoundError:
        return False
    except OSError as e:
        raise SessionError(f"cannot delete session {session_id!r}: {e}")


def resolve_from_args(args, command: str) -> Optional[Session]:
    """The session to resume from ``--resume``/``--continue``, or None for a new one.

    ``--resume <id|file>`` loads by id (in-place) or imports a file; ``--continue``
    picks the most recent session for `command`. Raises :class:`SessionError` on a
    bad id/file or an empty ``--continue`` -- callers map it to exit 2.
    """
    ref = getattr(args, "resume", None)
    if ref:
        return load(ref, command)
    if getattr(args, "cont", None):
        s = most_recent(command)
        if s is None:
            raise SessionError(f"no {command} session to --continue")
        return s
    return None


def apply_to_args(args, session: Optional[Session], command: str) -> None:
    """Restore a session's scalar settings onto `args`, explicit flags winning.

    Sets ``model``/``temperature``/``max_tokens``/``max_tool_calls`` **only where the
    raw arg is still None** -- so an explicit CLI flag wins, and (crucially) this runs
    BEFORE ``userconfig.apply_defaults`` (which also fills None dests), giving the
    precedence *explicit flag > saved session > config default*. ``venice_parameters``
    can't round-trip as scalar args, so it's restored later via :func:`merge_gen_kwargs`.
    """
    if session is None:
        return
    gk = session.gen_kwargs or {}
    if getattr(args, "model", None) is None and session.model:
        args.model = session.model
    if getattr(args, "temperature", None) is None and "temperature" in gk:
        args.temperature = gk["temperature"]
    if getattr(args, "max_tokens", None) is None and "max_tokens" in gk:
        args.max_tokens = gk["max_tokens"]
    if (getattr(args, "max_tool_calls", None) is None
            and session.max_tool_calls is not None):
        args.max_tool_calls = session.max_tool_calls


def merge_gen_kwargs(saved: Optional[dict], fresh: Optional[dict]) -> dict:
    """Merge a saved session's gen_kwargs under freshly-built ones (fresh wins).

    Top-level keys merge shallowly; ``extra_body.venice_parameters`` is deep-merged
    so a resumed chat keeps its saved web-search/character toggles unless the resume
    invocation re-specifies them. Scalar params (temperature/max_tokens) already flow
    through `args` via :func:`apply_to_args`; this is what restores the vp block.
    """
    saved = saved or {}
    fresh = fresh or {}
    merged = {**saved, **fresh}
    svp = (saved.get("extra_body") or {}).get("venice_parameters")
    fvp = (fresh.get("extra_body") or {}).get("venice_parameters")
    if isinstance(svp, dict) or isinstance(fvp, dict):
        eb = dict(merged.get("extra_body") or {})
        eb["venice_parameters"] = {**(svp or {}), **(fvp or {})}
        merged["extra_body"] = eb
    return merged
