"""Credential read/write. Never logs the key. Never prints the key.

The single Venice key lives in `credentials` (`load_key`/`save_key`). NAMED secrets
beyond it -- the embed-backend key, later MCP/cluster tokens (#43) -- live in a
structured 0600 `secrets.json` via `load_secret`/`save_secret`/`list_secrets`/
`delete_secret`. Same hygiene as the main key: values are never printed or logged;
`list_secrets` exposes lengths only.
"""
import getpass
import json
import os
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple

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


# --------------------------------------------------------------------------- #
# Named-secret store (secrets.json) -- #43
# --------------------------------------------------------------------------- #
# Names with a canonical env var: the env value wins over the stored one, so an
# ephemeral `$VENICE_EMBED_API_KEY` still overrides `secrets.json` (mirrors how
# `$VENICE_API_KEY` overrides the credentials file). Names not listed here are
# store-only (e.g. future MCP tokens).
_SECRET_ENV = {"embed": config.ENV_EMBED_API_KEY}

_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


def _valid_name(name: str) -> str:
    name = (name or "").strip()
    if not name or not _NAME_RE.match(name):
        raise AuthError(
            "secret name must be non-empty and use only letters, digits, '_', "
            f"'.', or '-' (got {name!r})"
        )
    return name


def _warn_loose_perms(p: Path) -> None:
    try:
        mode = p.stat().st_mode & 0o777
    except OSError:
        return
    if mode & 0o077:
        print(
            f"warning: {p} has loose permissions ({oct(mode)}); "
            f"run `chmod 600 {p}`",
            file=sys.stderr,
        )


def _load_secrets(*, strict: bool = False) -> dict:
    """Read secrets.json -> {"version":1,"secrets":{...}}.

    Tolerant by default (missing/corrupt -> empty doc + a one-line stderr warning),
    so a read never crashes a command. `strict=True` raises AuthError on a corrupt
    file so a mutating write never silently clobbers an unreadable store.
    """
    p: Path = config.SECRETS_FILE
    if not p.exists():
        return {"version": 1, "secrets": {}}
    _warn_loose_perms(p)
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(doc, dict) or not isinstance(doc.get("secrets"), dict):
            raise ValueError("not a secrets document")
    except (OSError, ValueError) as e:
        if strict:
            raise AuthError(
                f"{p} is unreadable or corrupt ({e}); fix or remove it before "
                "writing secrets."
            ) from None
        print(f"warning: ignoring unreadable {p} ({e})", file=sys.stderr)
        return {"version": 1, "secrets": {}}
    return doc


def _save_secrets(doc: dict) -> Path:
    """Atomically write secrets.json with mode 0600 (mirrors save_key)."""
    config.CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(config.CONFIG_DIR, 0o700)
    except OSError:
        pass
    tmp = config.SECRETS_FILE.with_suffix(".tmp")
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
    os.replace(tmp, config.SECRETS_FILE)
    try:
        os.chmod(config.SECRETS_FILE, 0o600)
    except OSError:
        pass
    return config.SECRETS_FILE


def save_secret(name: str, value: str) -> Path:
    """Store `value` under `name` in secrets.json (0600). Returns the path."""
    name = _valid_name(name)
    if not value or not value.strip():
        raise AuthError("Refusing to save an empty secret.")
    doc = _load_secrets(strict=True)
    doc["secrets"][name] = value
    return _save_secrets(doc)


def load_secret(name: str) -> Optional[str]:
    """Resolve a named secret: env var (if the name has one) > store > None.

    Returns None when unset -- callers decide whether that's an error (the embed
    key treats None as "no auth needed")."""
    env = _SECRET_ENV.get(name)
    if env:
        env_val = os.environ.get(env, "").strip()
        if env_val:
            return env_val
    val = _load_secrets()["secrets"].get(name)
    return val or None


def list_secrets() -> List[Tuple[str, int]]:
    """(name, value-length) pairs, sorted by name. Lengths only -- never values."""
    secrets = _load_secrets()["secrets"]
    return sorted((n, len(v)) for n, v in secrets.items())


def delete_secret(name: str) -> bool:
    """Remove a named secret. Returns True if it existed."""
    name = _valid_name(name)
    doc = _load_secrets(strict=True)
    if name not in doc["secrets"]:
        return False
    del doc["secrets"][name]
    _save_secrets(doc)
    return True


def prompt_and_save_secret(name: str) -> Path:
    """Interactive prompt (getpass, TTY-only) to store a named secret. No echo."""
    name = _valid_name(name)  # fail before prompting on a bad name
    if not sys.stdin.isatty():
        env = _SECRET_ENV.get(name)
        hint = f" Set ${env} instead." if env else ""
        raise AuthError(f"Setting a secret requires a TTY.{hint}")
    print(f"Enter the secret value for '{name}' (input is hidden).", file=sys.stderr)
    value = getpass.getpass(prompt=f"secret '{name}': ")
    path = save_secret(name, value)
    print(
        f"Saved {len(value)}-char secret '{name}' to {path} (mode 0600).",
        file=sys.stderr,
    )
    return path
