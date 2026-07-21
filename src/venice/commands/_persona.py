"""Local, file-backed system prompts ("personas") for `venice chat` (#68).

A persona is just a plain-text/markdown file under ``~/.config/venice/personas/``
whose contents become the session's system prompt -- the file-backed sibling of
inline ``/system``. This module is the one place that turns a bare persona *name*
into a file on disk, and it is deliberately small and paranoid because it decides
which files `venice chat` will read on the user's behalf.

Safety (the crux of the feature):

- Listing enumerates **only** ``PERSONAS_DIR`` -- never the config root, where the
  plaintext ``credentials`` file lives. A fun feature must never become a
  credential-enumeration path.
- A ``<name>`` is **bare-name only**. Anything with a path separator, a ``..``
  segment, or a name that isn't its own basename is rejected up front; the
  resolved file is then re-checked with :func:`_index.resolves_inside` (the
  repo's realpath containment primitive) so ``/persona ../credentials`` (or any
  symlink shenanigans) can't escape the personas dir.

Both guards are belt-and-suspenders on purpose: the string check is easy to read,
the realpath check is what actually holds under symlinks.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple

from .. import config
from . import _index

# Resolution order for a bare name: markdown first, then plain text.
_EXTENSIONS = (".md", ".txt")


class PersonaError(Exception):
    """A persona name is unsafe, or its file is missing/unreadable. Printable."""


def _reject_unsafe(name: str) -> None:
    """Raise unless `name` is a safe, bare persona name (no path traversal)."""
    if not name:
        raise PersonaError("no persona name given")
    if (
        "/" in name
        or "\\" in name
        or ".." in name
        or name != os.path.basename(name)
    ):
        raise PersonaError(f"invalid persona name {name!r} (bare names only)")


def resolve_path(name: str) -> Path:
    """Resolve a bare persona `name` to an existing file inside ``PERSONAS_DIR``.

    Rejects traversal, then returns the first of ``<name>.md`` / ``<name>.txt``
    that exists. Raises :class:`PersonaError` if the name is unsafe, the file is
    missing, or (defense in depth) the realpath escapes the personas dir.
    """
    _reject_unsafe(name)
    personas_dir = config.PERSONAS_DIR
    root = Path(os.path.realpath(personas_dir))
    for ext in _EXTENSIONS:
        candidate = personas_dir / (name + ext)
        real = Path(os.path.realpath(candidate))
        # Containment check first: a name that resolves outside the dir is unsafe
        # regardless of whether the file exists.
        if not _index.resolves_inside(real, root):
            raise PersonaError(f"persona {name!r} escapes {personas_dir}")
        if real.is_file():
            return candidate
    raise PersonaError(
        f"no persona {name!r} in {personas_dir} "
        f"(expected {name}.md or {name}.txt)"
    )


def load(name: str) -> str:
    """Return the system-prompt text for persona `name`.

    Raises :class:`PersonaError` on an unsafe name, a missing file, or a read
    error.
    """
    path = resolve_path(name)
    try:
        return path.read_text(encoding="utf-8")
    except OSError as e:
        raise PersonaError(f"cannot read persona {name!r}: {e}")


def _first_line(text: str) -> str:
    """First non-blank line, for the one-line description in the listing."""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def available() -> List[Tuple[str, str]]:
    """List personas as ``(name, first_line)`` pairs, sorted by name.

    Enumerates **only** ``PERSONAS_DIR`` (never the config root). A missing dir
    yields ``[]``. Unreadable individual files list with an empty description
    rather than aborting the whole listing.
    """
    personas_dir = config.PERSONAS_DIR
    seen: dict = {}
    try:
        entries = sorted(personas_dir.iterdir())
    except OSError:
        return []
    for entry in entries:
        if entry.suffix not in _EXTENSIONS or not entry.is_file():
            continue
        name = entry.stem
        # A name with both .md and .txt: .md wins (matches resolve order); don't
        # let the .txt overwrite it.
        if name in seen and entry.suffix != _EXTENSIONS[0]:
            continue
        try:
            desc = _first_line(entry.read_text(encoding="utf-8"))
        except OSError:
            desc = ""
        seen[name] = desc
    return sorted(seen.items())
