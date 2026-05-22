"""Detect and invoke a local audio player. Synchronous; non-fatal on failure."""
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable, List, Optional, Tuple

_PLAYERS: List[Tuple[str, Callable[[Path], List[str]]]] = [
    ("paplay", lambda p: ["paplay", str(p)]),
    ("aplay", lambda p: ["aplay", "-q", str(p)]),
    ("ffplay", lambda p: ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", str(p)]),
    ("mpg123", lambda p: ["mpg123", "-q", str(p)]),
    ("play", lambda p: ["play", "-q", str(p)]),
    ("afplay", lambda p: ["afplay", str(p)]),
]


def find_player() -> Optional[Tuple[str, Callable[[Path], List[str]]]]:
    for name, builder in _PLAYERS:
        if shutil.which(name):
            return name, builder
    return None


def has_player() -> bool:
    return find_player() is not None


def play(path: Path) -> bool:
    """Play synchronously. Returns True on exit-0, False otherwise.

    Never raises. Prints a hint to stderr if no player exists.
    """
    found = find_player()
    if not found:
        print(
            f"no audio player found (tried paplay/aplay/ffplay/mpg123/play/afplay)\n"
            f"saved to {path}; play manually with your preferred tool.",
            file=sys.stderr,
        )
        return False
    name, build = found
    try:
        completed = subprocess.run(
            build(path),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if completed.returncode != 0:
            print(
                f"{name} exited {completed.returncode}; saved to {path}",
                file=sys.stderr,
            )
            return False
        return True
    except (OSError, KeyboardInterrupt) as e:
        print(f"playback aborted ({e}); file is at {path}", file=sys.stderr)
        return False
