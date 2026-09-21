"""Disk-backed exact archive for evidence-preserving compaction (#74/#241)."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


MAX_ARCHIVE_BYTES = 1024 * 1024 * 1024
DEFAULT_ARCHIVE_MAX_MIB = MAX_ARCHIVE_BYTES // (1024 * 1024)
MAX_ARCHIVE_ENTRIES = 8192
MAX_ARCHIVE_METADATA_BYTES = 8 * 1024 * 1024
MAX_LIST_ENTRIES = 50
MAX_READ_BYTES = 32 * 1024
LIVE_INDEX_ENTRIES = 32
EXCERPT_CHARS = 160

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_ENCODER = json.JSONEncoder(
    ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False,
)


class ArchiveError(ValueError):
    """An archive envelope is malformed or a storage boundary was crossed."""


def max_bytes_from_args(args) -> int:
    value = getattr(args, "context_archive_max_mib", None)
    if value is None:
        value = DEFAULT_ARCHIVE_MAX_MIB
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ArchiveError("--context-archive-max-mib must be a positive integer")
    return value * 1024 * 1024


def positive_mib(value) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        raise ValueError("must be a positive integer") from None
    if isinstance(value, bool) or parsed <= 0:
        raise ValueError("must be a positive integer")
    return parsed


def _canonical(message: dict) -> str:
    try:
        return _ENCODER.encode(message)
    except (TypeError, ValueError) as e:
        raise ArchiveError(f"archived message is not finite JSON: {e}") from None


def _excerpt(text: str, *, tail: bool = False) -> str:
    value = text[-EXCERPT_CHARS:] if tail else text[:EXCERPT_CHARS]
    return value.replace("\n", "\\n")


def _metadata(entry: dict) -> dict:
    result = {
        "id": entry["id"],
        "role": entry["role"],
        "chars": entry["chars"],
        "bytes": entry["bytes"],
        "sha256": entry["sha256"],
        "blob": entry["blob"],
        "head": entry["head"],
        "tail": entry["tail"],
    }
    if entry.get("name"):
        result["name"] = entry["name"]
    return result


def _bounded_utf8_slice(text: str, offset: int, limit: int = MAX_READ_BYTES) -> str:
    """Return the longest char-aligned prefix no larger than ``limit`` bytes."""
    lo, hi = offset, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if len(text[offset:mid].encode("utf-8")) <= limit:
            lo = mid
        else:
            hi = mid - 1
    return text[offset:lo]


def _ensure_private_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    if path.is_symlink() or not path.is_dir():
        raise ArchiveError(f"context archive path is not a directory: {path}")
    try:
        os.chmod(path, 0o700)
    except OSError:
        pass


def _regular_file_size(path: Path) -> int:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(str(path), flags)
    except OSError as e:
        raise ArchiveError(f"cannot read context archive blob {path.name}: {e}") from None
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise ArchiveError(f"context archive blob {path.name} is not a regular file")
        return info.st_size
    finally:
        os.close(fd)


@dataclass
class ContextArchive:
    """Exact archived messages with bounded metadata and disk-backed bodies."""

    entries: List[dict] = field(default_factory=list)
    last_error: Optional[str] = None
    source_dir: Optional[Path] = field(default=None, repr=False)
    storage_dir: Optional[Path] = field(default=None, repr=False)
    max_bytes: Optional[int] = field(default=None, repr=False)
    _temporary: Optional[Path] = field(
        default=None, init=False, repr=False,
    )
    _verified: set = field(default_factory=set, init=False, repr=False)

    @classmethod
    def from_envelope(
        cls, raw, *, source_dir=None, max_bytes: Optional[int] = None,
    ) -> "ContextArchive":
        if raw is None:
            return cls(source_dir=Path(source_dir) if source_dir else None,
                       max_bytes=max_bytes)
        if not isinstance(raw, list):
            raise ArchiveError("context_archive must be a JSON list")
        if len(raw) > MAX_ARCHIVE_ENTRIES:
            raise ArchiveError(
                f"context_archive exceeds {MAX_ARCHIVE_ENTRIES} entries"
            )
        validated = []
        seen = set()
        for index, item in enumerate(raw, 1):
            if not isinstance(item, dict):
                raise ArchiveError(f"context_archive entry {index} is malformed")
            entry_id = item.get("id")
            expected_id = f"ctx-{index:06d}"
            if entry_id != expected_id:
                raise ArchiveError(f"context_archive entry {index} has an invalid id")
            if entry_id in seen:
                raise ArchiveError(f"context_archive entry {index} repeats id {entry_id}")
            seen.add(entry_id)

            message = item.get("message")
            if isinstance(message, dict):
                if not isinstance(message.get("role"), str):
                    raise ArchiveError(f"context_archive entry {index} has no role")
                text = _canonical(message)
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
                expected["message"] = copy.deepcopy(message)
                validated.append(expected)
                continue

            required = ("role", "chars", "bytes", "sha256", "blob", "head", "tail")
            if any(key not in item for key in required):
                raise ArchiveError(f"context_archive entry {index} is malformed")
            sha = item.get("sha256")
            if not isinstance(sha, str) or not _SHA256_RE.fullmatch(sha):
                raise ArchiveError(f"context_archive entry {entry_id} has invalid sha256")
            if item.get("blob") != f"{sha}.json":
                raise ArchiveError(f"context_archive entry {entry_id} has invalid blob")
            if not isinstance(item.get("role"), str):
                raise ArchiveError(f"context_archive entry {entry_id} has invalid role")
            for key in ("chars", "bytes"):
                value = item.get(key)
                if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                    raise ArchiveError(
                        f"context_archive entry {entry_id} has invalid {key}"
                    )
            for key in ("head", "tail"):
                if not isinstance(item.get(key), str):
                    raise ArchiveError(
                        f"context_archive entry {entry_id} has invalid {key}"
                    )
            if "name" in item and not isinstance(item.get("name"), str):
                raise ArchiveError(f"context_archive entry {entry_id} has invalid name")
            validated.append(_metadata(item))

        total = len(_canonical(validated).encode("utf-8"))
        if total > MAX_ARCHIVE_METADATA_BYTES:
            raise ArchiveError(
                f"context_archive metadata exceeds {MAX_ARCHIVE_METADATA_BYTES} bytes"
            )
        return cls(
            validated,
            source_dir=Path(source_dir) if source_dir else None,
            max_bytes=max_bytes,
        )

    @staticmethod
    def _entry(entry_id: str, message: dict, text: Optional[str] = None) -> dict:
        text = _canonical(message) if text is None else text
        raw = text.encode("utf-8")
        sha = hashlib.sha256(raw).hexdigest()
        entry = {
            "id": entry_id,
            "role": message["role"],
            "chars": len(text),
            "bytes": len(raw),
            "sha256": sha,
            "blob": f"{sha}.json",
            "head": _excerpt(text),
            "tail": _excerpt(text, tail=True),
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
    def byte_limit(self) -> int:
        return self.max_bytes if self.max_bytes is not None else MAX_ARCHIVE_BYTES

    @property
    def bytes_used(self) -> int:
        unique = {}
        for entry in self.entries:
            unique.setdefault(entry["sha256"], entry["bytes"])
        return sum(unique.values())

    def _ensure_bound(self) -> None:
        if self.storage_dir is not None:
            return
        if self.source_dir is not None:
            self.bind(self.source_dir)
            return
        self._temporary = Path(tempfile.mkdtemp(prefix="venice-context-archive-"))
        self.bind(self._temporary)

    def bind_temporary(self, *, max_bytes: Optional[int] = None) -> None:
        """Use disposable storage without mutating a resumed session sidecar."""
        self._temporary = Path(tempfile.mkdtemp(prefix="venice-context-archive-"))
        self.bind(
            self._temporary, max_bytes=max_bytes, copy_existing=False,
        )

    def _blob_path(self, entry: dict) -> Path:
        if self.storage_dir is not None:
            target = self.storage_dir / entry["blob"]
            if target.exists():
                return target
        if self.source_dir is not None:
            return self.source_dir / entry["blob"]
        raise ArchiveError("context archive is not bound to storage")

    def bind(self, directory, *, max_bytes: Optional[int] = None,
             copy_existing: bool = True) -> None:
        """Bind to private storage, migrating v2/source entries transactionally."""
        if max_bytes is not None:
            if not isinstance(max_bytes, int) or isinstance(max_bytes, bool) or max_bytes <= 0:
                raise ArchiveError("context archive byte limit must be a positive integer")
            self.max_bytes = max_bytes
        target = Path(directory)
        _ensure_private_dir(target)
        logical = self.bytes_used
        if logical > self.byte_limit:
            raise ArchiveError(
                f"evidence archive full: archive uses {logical} bytes "
                f"(limit {self.byte_limit}); increase --context-archive-max-mib"
            )

        created = []
        clean_entries = []
        try:
            for entry in self.entries:
                clean = _metadata(entry)
                destination = target / clean["blob"]
                if "message" in entry:
                    existed = destination.exists()
                    self._write_message_blob(destination, entry["message"], clean)
                    if not existed:
                        created.append(destination)
                elif self.source_dir is None:
                    raise ArchiveError(
                        f"context archive blob source is unavailable for {entry['id']}"
                    )
                elif self.source_dir == target or not copy_existing:
                    source = self.source_dir / clean["blob"]
                    size = _regular_file_size(source)
                    if size != clean["bytes"]:
                        raise ArchiveError(
                            f"context archive blob {clean['blob']} has invalid size"
                        )
                else:
                    source = self.source_dir / clean["blob"]
                    size = _regular_file_size(source)
                    if size != clean["bytes"]:
                        raise ArchiveError(
                            f"context archive blob {clean['blob']} has invalid size"
                        )
                    if not destination.exists():
                        tmp = target / (".migrate-" + clean["blob"])
                        shutil.copyfile(source, tmp)
                        os.chmod(tmp, 0o600)
                        os.replace(tmp, destination)
                        created.append(destination)
                clean_entries.append(clean)
        except (OSError, ArchiveError) as e:
            for path in created:
                try:
                    path.unlink()
                except OSError:
                    pass
            if isinstance(e, ArchiveError):
                raise
            raise ArchiveError(f"cannot bind context archive: {e}") from None

        self.entries = clean_entries
        self.storage_dir = target
        if copy_existing:
            self.source_dir = target
        self.last_error = None

    @staticmethod
    def _write_message_blob(path: Path, message: dict, expected: dict) -> None:
        if path.exists():
            if _regular_file_size(path) != expected["bytes"]:
                raise ArchiveError(f"context archive blob {path.name} has invalid size")
            return
        tmp = path.with_name(".write-" + path.name)
        fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            digest = hashlib.sha256()
            chars = byte_count = 0
            head = ""
            tail = ""
            with os.fdopen(fd, "wb") as stream:
                for piece in _ENCODER.iterencode(message):
                    raw = piece.encode("utf-8")
                    stream.write(raw)
                    digest.update(raw)
                    chars += len(piece)
                    byte_count += len(raw)
                    if len(head) < EXCERPT_CHARS:
                        head += piece[:EXCERPT_CHARS - len(head)]
                    tail = (tail + piece)[-EXCERPT_CHARS:]
                stream.flush()
                os.fsync(stream.fileno())
            observed = {
                "sha256": digest.hexdigest(), "chars": chars, "bytes": byte_count,
                "head": _excerpt(head), "tail": _excerpt(tail, tail=True),
            }
            for key, value in observed.items():
                if value != expected[key]:
                    raise ArchiveError(
                        f"context archive staging produced invalid {key}"
                    )
            os.replace(tmp, path)
            try:
                os.chmod(path, 0o600)
            except OSError:
                pass
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    @classmethod
    def _write_staged_message(cls, path: Path, entry_id: str, message: dict) -> dict:
        """Stream one canonical message to staging and return its metadata."""
        fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        digest = hashlib.sha256()
        chars = byte_count = 0
        head = ""
        tail = ""
        try:
            with os.fdopen(fd, "wb") as stream:
                try:
                    pieces = _ENCODER.iterencode(message)
                    for piece in pieces:
                        raw = piece.encode("utf-8")
                        stream.write(raw)
                        digest.update(raw)
                        chars += len(piece)
                        byte_count += len(raw)
                        if len(head) < EXCERPT_CHARS:
                            head += piece[:EXCERPT_CHARS - len(head)]
                        tail = (tail + piece)[-EXCERPT_CHARS:]
                except (TypeError, ValueError) as e:
                    raise ArchiveError(
                        f"archived message is not finite JSON: {e}"
                    ) from None
                stream.flush()
                os.fsync(stream.fileno())
        except Exception:
            try:
                path.unlink()
            except OSError:
                pass
            raise
        sha = digest.hexdigest()
        entry = {
            "id": entry_id,
            "role": message["role"],
            "chars": chars,
            "bytes": byte_count,
            "sha256": sha,
            "blob": f"{sha}.json",
            "head": _excerpt(head),
            "tail": _excerpt(tail, tail=True),
            "_stage_path": str(path),
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

    def stage(self, messages: List[dict]) -> List[dict]:
        """Write staged blobs and enforce caps without changing archive entries."""
        self._ensure_bound()
        staged = []
        try:
            for message in messages:
                if not isinstance(message, dict) or not isinstance(message.get("role"), str):
                    raise ArchiveError("message selected for archival is malformed")
                entry_id = f"ctx-{len(self.entries) + len(staged) + 1:06d}"
                stage_path = self.storage_dir / (
                    f".stage-{entry_id}-{os.urandom(6).hex()}.json"
                )
                entry = self._write_staged_message(stage_path, entry_id, message)
                staged.append(entry)

            count = len(self.entries) + len(staged)
            if count > MAX_ARCHIVE_ENTRIES:
                raise ArchiveError(
                    f"evidence archive full: compaction needs {count} entries "
                    f"(limit {MAX_ARCHIVE_ENTRIES})"
                )
            unique = {e["sha256"]: e["bytes"] for e in self.entries}
            for entry in staged:
                unique.setdefault(entry["sha256"], entry["bytes"])
            used = sum(unique.values())
            if used > self.byte_limit:
                raise ArchiveError(
                    f"evidence archive full: compaction needs {used} bytes "
                    f"(limit {self.byte_limit}); increase "
                    "--context-archive-max-mib or reset/start a new session"
                )
            metadata = [_metadata(e) for e in self.entries + staged]
            if len(_canonical(metadata).encode("utf-8")) > MAX_ARCHIVE_METADATA_BYTES:
                raise ArchiveError(
                    "evidence archive metadata is full; reset/start a new session"
                )
            return staged
        except Exception:
            self.discard(staged)
            raise

    def discard(self, staged: List[dict]) -> None:
        for entry in staged:
            raw = entry.get("_stage_path")
            if raw:
                try:
                    Path(raw).unlink()
                except OSError:
                    pass

    def commit(self, staged: List[dict]) -> None:
        clean = []
        try:
            for entry in staged:
                stage_path = Path(entry["_stage_path"])
                target = self.storage_dir / entry["blob"]
                if target.exists():
                    if _regular_file_size(target) != entry["bytes"]:
                        raise ArchiveError(
                            f"context archive blob {target.name} has invalid size"
                        )
                    stage_path.unlink()
                else:
                    os.replace(stage_path, target)
                    try:
                        os.chmod(target, 0o600)
                    except OSError:
                        pass
                clean.append(_metadata(entry))
        except (OSError, ArchiveError) as e:
            self.discard(staged)
            if isinstance(e, ArchiveError):
                raise
            raise ArchiveError(f"cannot commit context archive: {e}") from None
        self.entries.extend(clean)
        self.last_error = None

    def clear(self) -> None:
        if self.storage_dir is not None:
            for blob in {e.get("blob") for e in self.entries if e.get("blob")}:
                try:
                    (self.storage_dir / blob).unlink()
                except (FileNotFoundError, OSError):
                    pass
        self.entries.clear()
        self._verified.clear()
        self.last_error = None

    def close(self) -> None:
        if self._temporary is not None:
            shutil.rmtree(self._temporary, ignore_errors=True)
            self._temporary = None

    def __del__(self):  # pragma: no cover - explicit close paths are preferred
        try:
            self.close()
        except Exception:
            pass

    def to_envelope(self) -> list:
        if any("message" in entry for entry in self.entries):
            return copy.deepcopy(self.entries)
        return [_metadata(entry) for entry in self.entries]

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
            "byte_limit": self.byte_limit,
        }

    def _verify_blob(self, entry: dict) -> Path:
        path = self._blob_path(entry)
        if entry["sha256"] in self._verified:
            return path
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        try:
            fd = os.open(str(path), flags)
        except OSError as e:
            raise ArchiveError(
                f"cannot read context archive blob {path.name}: {e}"
            ) from None
        digest = hashlib.sha256()
        size = 0
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode):
                raise ArchiveError(
                    f"context archive blob {path.name} is not a regular file"
                )
            while True:
                chunk = os.read(fd, 1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
                size += len(chunk)
        finally:
            os.close(fd)
        if size != entry["bytes"] or digest.hexdigest() != entry["sha256"]:
            raise ArchiveError(f"context archive blob {path.name} failed verification")
        self._verified.add(entry["sha256"])
        return path

    def _read_slice(self, path: Path, offset: int) -> str:
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(str(path), flags)
        parts = []
        remaining_chars = offset
        remaining_bytes = MAX_READ_BYTES
        try:
            with os.fdopen(fd, "r", encoding="utf-8") as stream:
                while remaining_bytes > 0:
                    chunk = stream.read(16 * 1024)
                    if not chunk:
                        break
                    if remaining_chars:
                        if len(chunk) <= remaining_chars:
                            remaining_chars -= len(chunk)
                            continue
                        chunk = chunk[remaining_chars:]
                        remaining_chars = 0
                    piece = _bounded_utf8_slice(chunk, 0, remaining_bytes)
                    parts.append(piece)
                    remaining_bytes -= len(piece.encode("utf-8"))
                    if len(piece) < len(chunk):
                        break
        except UnicodeDecodeError as e:
            raise ArchiveError(f"context archive blob {path.name} is not UTF-8: {e}") from None
        return "".join(parts)

    def read(self, entry_id: str, offset: int = 0) -> dict:
        if not isinstance(entry_id, str):
            raise ArchiveError("entry_id must be a string")
        if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
            raise ArchiveError("offset must be a non-negative integer")
        entry = next((e for e in self.entries if e["id"] == entry_id), None)
        if entry is None:
            raise ArchiveError(f"unknown archive entry {entry_id!r}")
        if offset > entry["chars"]:
            raise ArchiveError(
                f"offset {offset} exceeds entry length {entry['chars']}"
            )
        if "message" in entry:
            text = _canonical(entry["message"])
            content = _bounded_utf8_slice(text, offset)
        else:
            content = self._read_slice(self._verify_blob(entry), offset)
        end = offset + len(content)
        return {
            "id": entry_id,
            "sha256": entry["sha256"],
            "offset": offset,
            "content": content,
            "next_offset": end if end < entry["chars"] else None,
            "complete": end >= entry["chars"],
            "chars": entry["chars"],
        }

    def live_index_message(self, staged: Optional[List[dict]] = None) -> dict:
        all_entries = self.entries + list(staged or [])
        newest = all_entries[-LIVE_INDEX_ENTRIES:]
        unique = {e["sha256"]: e["bytes"] for e in all_entries}
        used = sum(unique.values())
        lines = [
            "[Archived context evidence index]",
            f"{len(all_entries)} exact message(s), {used} byte(s) on disk. "
            "Use venice_context_archive to list metadata or read exact content.",
        ]
        older = len(all_entries) - len(newest)
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
