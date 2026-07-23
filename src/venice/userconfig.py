"""Persistent, non-secret user config at ~/.config/venice/config.json.

Two things live here:

- ``mcpServers`` -- an MCP server registry (like ``claude mcp add``) that the
  ``venice chat --mcp`` external-MCP client (#21) will load. (The built-in
  tool-calling loop, #15, is in-process and needs no registry.)
- ``defaults`` -- config-backed default flag values so users stop repeating
  ``--model`` / ``-o`` / ``--yes`` / ``--max-spend`` on every call (#17).

Precedence for a flag is CLI > env > config file > argparse default; this module
owns only the "config file" layer. The API key NEVER lives here -- it stays in
``credentials`` (see auth.py). The file is written mode 0600 because an MCP
``env``/``headers`` entry can carry a bearer token.
"""
import inspect
import json
import os
import sys
from pathlib import Path

from . import config


class ConfigError(Exception):
    """Config file present but unusable. Message is safe to print."""


def _default_doc() -> dict:
    """A fresh, empty config document (never share a mutable literal)."""
    return {"version": 1, "mcpServers": {}, "defaults": {}}


# --------------------------------------------------------------------------- #
# Read / write
# --------------------------------------------------------------------------- #
def load_config() -> dict:
    """Read config.json tolerantly. Missing or malformed -> a fresh default doc
    plus a one-line stderr warning. NEVER raises -- this runs at the top of every
    command, so a broken file must degrade to "no defaults", not a crash."""
    p = config.CONFIG_FILE
    if not p.exists():
        return _default_doc()
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        print(f"warning: ignoring unreadable {p}: {e}", file=sys.stderr)
        return _default_doc()
    if not isinstance(doc, dict):
        print(f"warning: ignoring {p}: top level is not a JSON object", file=sys.stderr)
        return _default_doc()
    return doc


def load_config_for_write() -> dict:
    """Like load_config, but raise ConfigError on a present-but-malformed file so
    a mutating command never silently clobbers user data. Absent -> fresh doc."""
    p = config.CONFIG_FILE
    if not p.exists():
        return _default_doc()
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise ConfigError(f"{p} is unreadable ({e}); fix or remove it first") from None
    if not isinstance(doc, dict):
        raise ConfigError(f"{p} is not a JSON object; fix or remove it first")
    return doc


def save_config(doc: dict) -> Path:
    """Atomically write config.json with mode 0600 (mirrors auth.save_key).
    Raises OSError on a disk failure (callers map that to exit 9)."""
    config.CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(config.CONFIG_DIR, 0o700)
    except OSError:
        pass

    tmp = config.CONFIG_FILE.with_suffix(".tmp")
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
    os.replace(tmp, config.CONFIG_FILE)
    try:
        os.chmod(config.CONFIG_FILE, 0o600)
    except OSError:
        pass
    return config.CONFIG_FILE


# --------------------------------------------------------------------------- #
# Dotted-key access (mutates the loaded doc in place; unknown keys survive a
# round-trip because save_config writes the whole doc back).
# --------------------------------------------------------------------------- #
def get_value(doc: dict, dotted: str):
    """Nested lookup by dotted key (e.g. "defaults.chat.model"). KeyError if absent."""
    node = doc
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            raise KeyError(dotted)
        node = node[part]
    return node


def set_value(doc: dict, dotted: str, value) -> None:
    """Set a dotted key, creating intermediate tables. ConfigError if an
    intermediate key exists but is not a table."""
    parts = dotted.split(".")
    node = doc
    for part in parts[:-1]:
        nxt = node.get(part)
        if nxt is None:
            nxt = {}
            node[part] = nxt
        elif not isinstance(nxt, dict):
            raise ConfigError(f"cannot set {dotted!r}: {part!r} is not a table")
        node = nxt
    node[parts[-1]] = value


def unset_value(doc: dict, dotted: str) -> bool:
    """Delete a dotted key. Returns True if it existed, False otherwise."""
    parts = dotted.split(".")
    node = doc
    for part in parts[:-1]:
        if not isinstance(node, dict) or part not in node:
            return False
        node = node[part]
    if isinstance(node, dict) and parts[-1] in node:
        del node[parts[-1]]
        return True
    return False


# --------------------------------------------------------------------------- #
# MCP server registry helpers
# --------------------------------------------------------------------------- #
def mcp_map(doc: dict) -> dict:
    m = doc.get("mcpServers")
    return m if isinstance(m, dict) else {}


def mcp_get(doc: dict, name: str):
    return mcp_map(doc).get(name)


def mcp_add(doc: dict, name: str, entry: dict) -> None:
    servers = doc.get("mcpServers")
    if not isinstance(servers, dict):
        servers = {}
        doc["mcpServers"] = servers
    servers[name] = entry


def mcp_remove(doc: dict, name: str) -> bool:
    servers = doc.get("mcpServers")
    if isinstance(servers, dict) and name in servers:
        del servers[name]
        return True
    return False


# --------------------------------------------------------------------------- #
# Shell allow/deny policy (issue #33)
# --------------------------------------------------------------------------- #
def shell_policy(doc: dict) -> dict:
    """Read the top-level ``shell`` policy section (single source of truth for both
    `venice chat --shell` and `venice code`'s `run` tool). Returns
    ``{"allow": [...], "deny": [...]}`` -- string lists, empty when unset/malformed.

    Mirrors :func:`mcp_map`: a non-`defaults` top-level section with its own reader
    (it isn't a per-command preference, so it doesn't flow through `apply_defaults`).
    ``venice config set shell.deny '["rm *"]'`` round-trips through the generic
    dotted-key store with no extra plumbing.
    """
    section = doc.get("shell")
    if not isinstance(section, dict):
        return {"allow": [], "deny": []}
    return {
        "allow": _as_list(section["allow"]) if section.get("allow") else [],
        "deny": _as_list(section["deny"]) if section.get("deny") else [],
    }


def browser_policy(doc: dict) -> dict:
    """Read the top-level ``browser`` policy section: the URL allow/deny lists for the
    #71 ``web_fetch``/``browser_capture`` tools. Returns ``{"allow": [...], "deny": [...]}``
    -- string lists, empty when unset/malformed.

    Mirrors :func:`shell_policy` (a non-`defaults` top-level section with its own reader);
    ``venice config set browser.deny '["*.internal"]'`` round-trips through the generic
    dotted-key store with no extra plumbing. The hardcoded stops (http/https only, cloud
    metadata blocked) live in ``_browser.check_url_policy`` and are not configurable.
    """
    section = doc.get("browser")
    if not isinstance(section, dict):
        return {"allow": [], "deny": []}
    return {
        "allow": _as_list(section["allow"]) if section.get("allow") else [],
        "deny": _as_list(section["deny"]) if section.get("deny") else [],
    }


# --------------------------------------------------------------------------- #
# #17 default-flag loader
# --------------------------------------------------------------------------- #
def _as_path(v):
    return Path(str(v)).expanduser()


def _as_bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "on")
    return bool(v)


def _as_list(v):
    """A config default that feeds an ``action="append"`` flag: pass a JSON list
    through, wrap a bare string as a single-element list."""
    if isinstance(v, list):
        return [str(x) for x in v]
    return [str(v)]


# config key -> (argparse dest, coercer). Globals apply to any command that
# declares the flag; a per-command section overrides them.
_GLOBAL_MAP = {
    "output_dir": ("output", _as_path),
    "max_spend": ("max_spend", float),
    "yes": ("yes", _as_bool),
}
_COMMAND_MAP = {
    "chat": {
        "model": ("model", str),
        "system": ("system", str),
        "persona": ("persona", str),
        "temperature": ("temperature", float),
        "max_tokens": ("max_tokens", int),
        "web_search": ("web_search", str),
        "character": ("character", str),
        "tools": ("tools", _as_bool),
        "max_tool_calls": ("max_tool_calls", int),
        "mcp": ("mcp", _as_list),
        "auto_compact": ("auto_compact", _as_bool),
        "compact_threshold": ("compact_threshold", int),
        "compact_keep_turns": ("compact_keep_turns", int),
        "session_max_spend": ("session_max_spend", float),
    },
    "embed": {
        "model": ("model", str),
        "dimensions": ("dimensions", int),
        "encoding_format": ("encoding_format", str),
        "embed_base_url": ("embed_base_url", str),
        "embed_model": ("embed_model", str),
        "embed_ca_bundle": ("embed_ca_bundle", str),
    },
    "index": {
        "model": ("model", str),
        "dimensions": ("dimensions", int),
        "embed_base_url": ("embed_base_url", str),
        "embed_model": ("embed_model", str),
        "embed_ca_bundle": ("embed_ca_bundle", str),
        "batch": ("batch", int),
        "chunk_lines": ("chunk_lines", int),
        "chunk_overlap": ("chunk_overlap", int),
        "exclude": ("exclude", _as_list),
    },
    "search": {
        "top_k": ("top_k", int),
        "embed_ca_bundle": ("embed_ca_bundle", str),
    },
    "code": {
        "model": ("model", str),
        "system": ("system", str),
        "root": ("root", str),
        "auto": ("auto", _as_bool),
        "assets": ("assets", _as_bool),
        "scout": ("scout", _as_bool),  # #52: opt-in read-only scout subagent
        "max_tool_calls": ("max_tool_calls", int),
        "exec_timeout": ("exec_timeout", int),
        "auto_compact": ("auto_compact", _as_bool),
        "compact_threshold": ("compact_threshold", int),
        "compact_keep_turns": ("compact_keep_turns", int),
        "session_max_spend": ("session_max_spend", float),
    },
    "image": {
        # `--hide-watermark` / `--safe-mode` are tri-state (default None) so these
        # defaults can win; an explicit CLI flag still wins over config.
        "hide_watermark": ("hide_watermark", _as_bool),
        "safe_mode": ("safe_mode", _as_bool),
        # Sizing / style / passthrough knobs (all default None on the CLI).
        "width": ("width", int),
        "height": ("height", int),
        "aspect_ratio": ("aspect_ratio", str),
        "resolution": ("resolution", str),
        "style_prefix": ("style_prefix", str),
        "preset": ("preset", str),
        "preset_file": ("preset_file", _as_path),
        "negative_prompt": ("negative_prompt", str),
        "cfg_scale": ("cfg_scale", float),
        "steps": ("steps", int),
        "style_preset": ("style_preset", str),
    },
    "image_edit": {
        "model": ("model", str),
        "aspect_ratio": ("aspect_ratio", str),
        "resolution": ("resolution", str),
        "output_format": ("output_format", str),
    },
    "tts": {
        "voice": ("voice", str),
        "speed": ("speed", float),
        # `--play`/`--no-play` is a tri-stated store_true(None)/store_false pair.
        "play": ("play", _as_bool),
    },
    "sfx": {
        "play": ("play", _as_bool),
    },
    "music": {
        # `lyrics` is deliberately CLI-only -- it's per-song content, not a
        # persistent preference.
        "duration": ("duration", int),
        "speed": ("speed", float),
        "play": ("play", _as_bool),
    },
    "video": {
        "model": ("model", str),
        "resolution": ("resolution", str),
        "aspect_ratio": ("aspect_ratio", str),
        "negative_prompt": ("negative_prompt", str),
    },
    "upscale": {
        "enhance_creativity": ("enhance_creativity", float),
        "enhance_prompt": ("enhance_prompt", str),
        "replication": ("replication", float),
    },
    # #71 browser tools -- safe knobs only. The URL allow/deny policy is NOT here: it must
    # never be model-overridable, so it flows through `browser_policy` (like `shell_policy`),
    # not the tool-argument merge path. `config_defaults_for` injects only the keys each
    # impl accepts (web_fetch: max_bytes/timeout; browser_capture: wait_ms/timeout).
    "browser": {
        "wait_ms": ("wait_ms", int),
        "timeout": ("timeout", int),
        "max_bytes": ("max_bytes", int),
    },
}


def resolve_default(command: str, key: str, doc=None):
    """Value for a defaults key, per-command section overriding a global scalar.
    None if unset. `key` is the config key (e.g. "model", "output_dir")."""
    if doc is None:
        doc = load_config()
    defaults = doc.get("defaults")
    if not isinstance(defaults, dict):
        return None
    section = defaults.get(command)
    if isinstance(section, dict) and key in section:
        return section[key]
    val = defaults.get(key)
    if isinstance(val, dict):  # a command section, not a global scalar
        return None
    return val


def config_defaults_for(section: str, impl, doc=None) -> dict:
    """Config-backed defaults for a tool `impl`, as a kwargs dict (issue #58).

    Only keys in ``_COMMAND_MAP[section]`` (the #57 allow-list) whose ``dest`` the
    `impl` actually accepts are included; each value is coerced. ``doc=None`` (no
    config) or an unknown section yields ``{}``. Never raises -- a bad value is
    skipped so tool building can't be broken by config. Callers layer this UNDER a
    tool's explicit args (precedence: explicit arg > config default > impl hardcoded
    default), the tool-path analogue of the CLI-side :func:`apply_defaults`.
    """
    if doc is None:
        return {}
    section_map = _COMMAND_MAP.get(section)
    if not section_map:
        return {}
    try:
        params = set(inspect.signature(impl).parameters)
    except (TypeError, ValueError):
        return {}
    out: dict = {}
    for key, (dest, coerce) in section_map.items():
        if dest not in params:
            continue  # tool doesn't take this preference
        raw = resolve_default(section, key, doc)
        if raw is None:
            continue
        try:
            out[dest] = coerce(raw)
        except (TypeError, ValueError):
            pass  # a bad config value shouldn't break tool building
    return out


def apply_defaults(args, command: str, doc=None) -> None:
    """Fill config-backed defaults onto `args`, but only where the dest is still
    None (so an explicit CLI flag always wins -- mirrors image._resolve_preset).
    Never raises: a bad config value is warned about and skipped."""
    if doc is None:
        doc = load_config()
    mapping = dict(_GLOBAL_MAP)
    mapping.update(_COMMAND_MAP.get(command, {}))
    for key, (dest, coerce) in mapping.items():
        if not hasattr(args, dest):
            continue  # this command doesn't declare the flag
        if getattr(args, dest) is not None:
            continue  # CLI (or an earlier layer) already set it
        raw = resolve_default(command, key, doc)
        if raw is None:
            continue
        try:
            setattr(args, dest, coerce(raw))
        except (TypeError, ValueError):
            print(
                f"{command}: ignoring invalid config default {key}={raw!r}",
                file=sys.stderr,
            )
