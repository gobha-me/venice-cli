"""Per-session file mailbox for mid-run steering (#78).

A running ``venice code`` agent -- especially ``--auto`` -- has only two controls:
let it finish, or kill it (losing uncommitted work + metered spend). This module
is the transport that adds a third: *steer* it. ``venice sessions send`` drops a
message here; the agent drains it at the next checkpoint (the top of each turn in
:func:`_agent.run_loop`, between tool batches) and appends it as a tagged user
turn -- exactly as if the operator had typed it in ``--interactive``.

Design (stdlib-only; no sockets, daemon, or threads -- the mailbox is the whole
channel):

* One message == one file at ``<SESSIONS_DIR>/<id>/mailbox/<stamp>-<hex>.msg``,
  a sibling of the session envelope ``<SESSIONS_DIR>/<id>.json`` (which the store's
  ``*.json`` glob ignores, so the two never collide).
* Written with the store's 0700-dir / 0600-file hygiene via atomic rename
  (``.tmp`` -> ``.msg``), mirroring :func:`_session.save`.
* Consumed in ``(mtime, name)`` order and deleted on read; a queued message
  survives an agent restart -- ``--resume`` drains it on the loop's first turn.

Trust model (CLAUDE.md): a steer is *additive user input* at the same trust level
as the original task author. The mailbox is a local, owner-only directory, NOT a
remote-control channel; nothing here reads or writes the API key.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from . import _index, _session

_MSG_SUFFIX = ".msg"


def mailbox_dir(session_id: str) -> Path:
    """Resolve ``<zone>/<id>/mailbox`` for a safe bare id.

    Reuses the session store's id-safety (:func:`_session._reject_unsafe`) and the
    same realpath containment check as :func:`_session._resolve_zone_path`, so a
    crafted id can't escape the sessions zone via ``..`` or a symlink.
    """
    _session._reject_unsafe(session_id)
    zone = _session._sessions_dir()
    root = Path(os.path.realpath(zone))
    base = zone / session_id
    if not _index.resolves_inside(Path(os.path.realpath(base)), root):
        raise _session.SessionError(f"session id {session_id!r} escapes {zone}")
    return base / "mailbox"


def _msg_name() -> str:
    """A lexically time-ordered filename ``YYYYmmddTHHMMSS.ffffff-<6hex>.msg``.

    Microsecond stamp keeps ``(mtime, name)`` order stable for sends in the same
    second; the random suffix avoids a collision (mirrors :func:`_session.new_id`).
    """
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%f")
    return f"{stamp}-{os.urandom(3).hex()}{_MSG_SUFFIX}"


def deposit(session_id: str, text: str) -> Path:
    """Atomically write `text` as one 0600 message in the session's mailbox.

    mkdir 0700 -> tmp 0600 -> write/fsync -> atomic rename -> chmod 0600 (mirrors
    :func:`_session.save`). Returns the final path; raises OSError on disk failure.
    """
    box = mailbox_dir(session_id)
    box.mkdir(parents=True, exist_ok=True)
    # mkdir honors umask, so re-assert 0700 on both <id>/ and <id>/mailbox.
    for d in (box.parent, box):
        try:
            os.chmod(d, 0o700)
        except OSError:
            pass
    path = box / _msg_name()
    tmp = path.with_suffix(".tmp")
    fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
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
    os.replace(tmp, path)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    return path


def _iter_msgs(box: Path) -> List[Path]:
    """Message paths in ``(mtime, name)`` order; tolerate a missing dir / racing writer."""
    try:
        entries = [p for p in box.iterdir() if p.suffix == _MSG_SUFFIX and p.is_file()]
    except OSError:
        return []

    def _key(p: Path):
        try:
            mtime = p.stat().st_mtime
        except OSError:
            mtime = 0.0
        return (mtime, p.name)

    return sorted(entries, key=_key)


def pending(session_id: str) -> int:
    """Count queued messages without consuming them (for ``sessions show``/``ls``)."""
    try:
        box = mailbox_dir(session_id)
    except _session.SessionError:
        return 0
    return len(_iter_msgs(box))


def drain(session_id: str) -> List[str]:
    """Read + delete all queued messages, oldest first, and return their texts.

    A missing mailbox yields ``[]``. An unreadable message is skipped; an in-flight
    ``.tmp`` write is ignored (only ``.msg`` files are drained). Safe to call every
    turn -- an empty mailbox is a single ``iterdir``.
    """
    try:
        box = mailbox_dir(session_id)
    except _session.SessionError:
        return []
    out: List[str] = []
    for p in _iter_msgs(box):
        try:
            text = p.read_text(encoding="utf-8")
        except OSError:
            continue
        try:
            p.unlink()
        except OSError:
            pass
        out.append(text)
    return out
