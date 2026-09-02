"""Bounded, lossless local archive for evidence-preserving compaction (#74)."""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from typing import List, Optional


MAX_ARCHIVE_BYTES = 8 * 1024 * 1024
MAX_ARCHIVE_ENTRIES = 512
MAX_LIST_ENTRIES = 50
MAX_READ_BYTES = 32 * 1024
LIVE_INDEX_ENTRIES = 32
EXCERPT_CHARS = 160


class ArchiveError(ValueError):
    """An archive envelope is malformed or a capacity boundary was crossed."""


def _canonical(message: dict) -> str:
    try:
        return json.dumps(
            message, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as e:
        raise ArchiveError(f"archived message is not finite JSON: {e}") from None


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _excerpt(text: str, *, tail: bool = False) -> str:
    value = text[-EXCERPT_CHARS:] if tail else text[:EXCERPT_CHARS]
    return value.replace("\n", "\\n")


def _metadata(entry: dict) -> dict:
    result = {
        "id": entry["id"],
        "role": entry["role"],
        "chars": entry["chars"],
        "sha256": entry["sha256"],
        "head": entry["head"],
        "tail": entry["tail"],
    }
    if entry.get("name"):
        result["name"] = entry["name"]
    return result


def _bounded_utf8_slice(text: str, offset: int) -> str:
    """Return the longest char-aligned slice no larger than MAX_READ_BYTES."""
    lo, hi = offset, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if len(text[offset:mid].encode("utf-8")) <= MAX_READ_BYTES:
            lo = mid
        else:
            hi = mid - 1
    return text[offset:lo]


@dataclass
class ContextArchive:
    """A bounded collection of exact, canonicalized archived messages."""

    entries: List[dict] = field(default_factory=list)
    last_error: Optional[str] = None

    @classmethod
    def from_envelope(cls, raw) -> "ContextArchive":
        if raw is None:
            return cls()
        if not isinstance(raw, list):
            raise ArchiveError("context_archive must be a JSON list")
        if len(raw) > MAX_ARCHIVE_ENTRIES:
            raise ArchiveError(
                f"context_archive exceeds {MAX_ARCHIVE_ENTRIES} entries"
            )
        validated = []
        seen = set()
        for index, item in enumerate(raw, 1):
            if not isinstance(item, dict) or not isinstance(item.get("message"), dict):
                raise ArchiveError(f"context_archive entry {index} is malformed")
            message = item["message"]
            if not isinstance(message.get("role"), str):
                raise ArchiveError(f"context_archive entry {index} has no role")
            text = _canonical(message)
            entry_id = item.get("id")
            expected_id = f"ctx-{index:06d}"
            if entry_id != expected_id:
                raise ArchiveError(f"context_archive entry {index} has an invalid id")
            if entry_id in seen:
                raise ArchiveError(f"context_archive entry {index} repeats id {entry_id}")
            seen.add(entry_id)
            expected = cls._entry(entry_id, message, text)
            for key in ("sha256", "chars", "role", "head", "tail"):
                if item.get(key) != expected[key]:
                    raise ArchiveError(
                        f"context_archive entry {entry_id} has invalid {key}"
                    )
            if item.get("name") != expected.get("name"):
                raise ArchiveError(
                    f"context_archive entry {entry_id} has invalid name"
                )
            validated.append(expected)
        total = len(_canonical(validated).encode("utf-8"))
        if total > MAX_ARCHIVE_BYTES:
            raise ArchiveError(f"context_archive exceeds {MAX_ARCHIVE_BYTES} bytes")
        return cls(validated)

    @staticmethod
    def _entry(entry_id: str, message: dict, text: Optional[str] = None) -> dict:
        text = _canonical(message) if text is None else text
        entry = {
            "id": entry_id,
            "role": message["role"],
            "chars": len(text),
            "sha256": _digest(text),
            "head": _excerpt(text),
            "tail": _excerpt(text, tail=True),
            "message": copy.deepcopy(message),
        }
        name = message.get("name")
        if isinstance(name, str) and name:
            entry["name"] = name
        elif message.get("role") == "assistant":
            calls = message.get("tool_calls")
            if isinstance(calls, list):
                names = []
                for call in calls:
                    fn = call.get("function") if isinstance(call, dict) else None
                    if isinstance(fn, dict) and isinstance(fn.get("name"), str):
                        names.append(fn["name"])
                if names:
                    entry["name"] = ",".join(names)
        return entry

    @property
    def bytes_used(self) -> int:
        return len(_canonical(self.entries).encode("utf-8"))

    def stage(self, messages: List[dict]) -> List[dict]:
        """Build entries and enforce both caps without changing the archive."""
        staged = []
        for message in messages:
            if not isinstance(message, dict) or not isinstance(message.get("role"), str):
                raise ArchiveError("message selected for archival is malformed")
            text = _canonical(message)
            entry_id = f"ctx-{len(self.entries) + len(staged) + 1:06d}"
            staged.append(self._entry(entry_id, message, text))
        count = len(self.entries) + len(staged)
        if count > MAX_ARCHIVE_ENTRIES:
            raise ArchiveError(
                f"evidence archive full: compaction needs {count} entries "
                f"(limit {MAX_ARCHIVE_ENTRIES})"
            )
        used = len(_canonical(self.entries + staged).encode("utf-8"))
        if used > MAX_ARCHIVE_BYTES:
            raise ArchiveError(
                f"evidence archive full: compaction needs {used} bytes "
                f"(limit {MAX_ARCHIVE_BYTES})"
            )
        return staged

    def commit(self, staged: List[dict]) -> None:
        self.entries.extend(staged)
        self.last_error = None

    def clear(self) -> None:
        self.entries.clear()
        self.last_error = None

    def to_envelope(self) -> list:
        return copy.deepcopy(self.entries)

    def list_page(self, cursor: int = 0, limit: int = MAX_LIST_ENTRIES) -> dict:
        if not isinstance(cursor, int) or isinstance(cursor, bool) or cursor < 0:
            raise ArchiveError("cursor must be a non-negative integer")
        if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= MAX_LIST_ENTRIES:
            raise ArchiveError(f"limit must be between 1 and {MAX_LIST_ENTRIES}")
        page = self.entries[cursor:cursor + limit]
        end = cursor + len(page)
        return {
            "entries": [_metadata(e) for e in page],
            "cursor": cursor,
            "next_cursor": end if end < len(self.entries) else None,
            "total": len(self.entries),
            "bytes": self.bytes_used,
        }

    def read(self, entry_id: str, offset: int = 0) -> dict:
        if not isinstance(entry_id, str):
            raise ArchiveError("entry_id must be a string")
        if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
            raise ArchiveError("offset must be a non-negative integer")
        entry = next((e for e in self.entries if e["id"] == entry_id), None)
        if entry is None:
            raise ArchiveError(f"unknown archive entry {entry_id!r}")
        text = _canonical(entry["message"])
        if offset > len(text):
            raise ArchiveError(f"offset {offset} exceeds entry length {len(text)}")
        content = _bounded_utf8_slice(text, offset)
        end = offset + len(content)
        return {
            "id": entry_id,
            "sha256": entry["sha256"],
            "offset": offset,
            "content": content,
            "next_offset": end if end < len(text) else None,
            "complete": end >= len(text),
            "chars": len(text),
        }

    def live_index_message(self) -> dict:
        newest = self.entries[-LIVE_INDEX_ENTRIES:]
        lines = [
            "[Archived context evidence index]",
            f"{len(self.entries)} exact message(s), {self.bytes_used} byte(s). "
            "Use venice_context_archive to list metadata or read exact content.",
        ]
        older = len(self.entries) - len(newest)
        if older:
            lines.append(f"{older} older entry/entries are discoverable with list pagination.")
        for entry in newest:
            label = entry["role"]
            if entry.get("name"):
                label += f"/{entry['name']}"
            lines.append(
                f"{entry['id']} {label} {entry['chars']} chars "
                f"sha256={entry['sha256']} head={entry['head']!r} tail={entry['tail']!r}"
            )
        return {"role": "system", "content": "\n".join(lines)}


def archive_tool(archive: ContextArchive):
    """Build the current-session, read-only archive lookup tool."""
    from . import _agent

    def invoke(args, *, confirm=False):
        del confirm
        try:
            action = args.get("action")
            if action == "list":
                return {"status": "ok", **archive.list_page(
                    args.get("cursor", 0), args.get("limit", MAX_LIST_ENTRIES)
                )}
            if action == "read":
                return {"status": "ok", **archive.read(
                    args.get("entry_id"), args.get("offset", 0)
                )}
            return {"status": "error", "message": "action must be list or read"}
        except ArchiveError as e:
            return {"status": "error", "message": str(e)}

    return _agent.Tool(
        name="venice_context_archive",
        description=(
            "Read evidence archived from earlier context compactions in this session. "
            "List bounded metadata pages, then read exact canonical message JSON by id."
        ),
        parameters={
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["list", "read"]},
                "entry_id": {"type": "string"},
                "cursor": {"type": "integer", "minimum": 0},
                "offset": {"type": "integer", "minimum": 0},
                "limit": {"type": "integer", "minimum": 1, "maximum": MAX_LIST_ENTRIES},
            },
            "required": ["action"],
            "additionalProperties": False,
        },
        invoke=invoke,
        paid=False,
        category="context",
        tags=("read",),
    )
