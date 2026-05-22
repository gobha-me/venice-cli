"""Credential read/write. Never logs the key. Never prints the key."""
import getpass
import os
import sys
from pathlib import Path

from . import config


class AuthError(Exception):
    """Credential not found or unusable. Message is safe to print."""


def load_key() -> str:
    """Resolve API key with precedence: env var > credentials file > AuthError."""
    env_val = os.environ.get(config.ENV_API_KEY, "").strip()
    if env_val:
        return env_val

    p: Path = config.CREDS_FILE
    if not p.exists():
        raise AuthError(
            f"No API key found. Set ${config.ENV_API_KEY} or run: venice login"
        )

    try:
        mode = p.stat().st_mode & 0o777
        if mode & 0o077:
            print(
                f"warning: {p} has loose permissions ({oct(mode)}); "
                f"run `chmod 600 {p}`",
                file=sys.stderr,
            )
        key = p.read_text(encoding="utf-8").strip()
    except OSError as e:
        raise AuthError(f"Cannot read {p}: {e}") from None

    if not key:
        raise AuthError(f"{p} is empty. Run: venice login")
    return key


def save_key(key: str) -> Path:
    """Atomically write the key with mode 0600. Returns the path written."""
    key = (key or "").strip()
    if not key:
        raise AuthError("Refusing to save an empty key.")
    if any(c.isspace() for c in key):
        raise AuthError("Key contains whitespace; check what you pasted.")

    config.CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(config.CONFIG_DIR, 0o700)
    except OSError:
        pass

    tmp = config.CREDS_FILE.with_suffix(".tmp")
    fd = os.open(
        str(tmp),
        os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
        0o600,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(key + "\n")
            f.flush()
            os.fsync(f.fileno())
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    os.replace(tmp, config.CREDS_FILE)
    try:
        os.chmod(config.CREDS_FILE, 0o600)
    except OSError:
        pass
    return config.CREDS_FILE


def prompt_and_save() -> Path:
    """Interactive prompt used by `venice login`. Uses getpass -- no echo.

    Refuses non-TTY stdin so we don't fall back to cleartext input.
    """
    if not sys.stdin.isatty():
        raise AuthError(
            f"Interactive login requires a TTY. "
            f"Set ${config.ENV_API_KEY} in your environment instead."
        )

    print(
        "Paste your Venice API key (from https://venice.ai/settings/api).",
        file=sys.stderr,
    )
    print("Input is hidden; it will not appear on screen.", file=sys.stderr)
    key = getpass.getpass(prompt="API key: ")
    path = save_key(key)
    print(f"Saved {len(key)}-char key to {path} (mode 0600).", file=sys.stderr)
    return path
