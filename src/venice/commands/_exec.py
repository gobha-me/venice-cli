"""Shared shell/exec rails for `venice code`'s `run`/`git` tools and the
`venice chat --shell` tool (issue #33).

Extracted from `_code` so both surfaces share ONE gate: the Venice API keys
scrubbed from the child env, cwd forced to a root, a timeout, size-capped captured
output, the confirm gate, and the allow/deny policy. **stdlib only** -- this module
imports nothing from the package so it stays a dependency leaf (no import cycle with
`_agent`, which builds the chat `shell` Tool from these primitives).

Policy (issue #33 decision -- "simple-command + globs"):

- **Deny** globs (`fnmatch`) are matched against the whole command string AND each
  shell token (and its basename), and are ALWAYS enforced. Deny wins over allow.
  Use ``rm`` / ``sudo`` to block a command by name, ``*rm -rf*`` to block a
  substring anywhere in the line.
- **Allow**, when non-empty, additionally requires a *single simple command*: the
  line may contain no shell operators/pipes/redirects/substitutions/variables
  (``; | & < > ( ) ` $`` or a newline), and the leading token's basename must match
  an allow entry (globs allowed, e.g. ``git``). An empty allowlist = unrestricted
  (only the confirm gate + deny apply) -- today's `venice code` behavior.

The exec boundary is the confirm gate + cwd + timeout + env-scrub + this policy,
**not** path containment: a shell command can still read/write outside the root,
which is why it is always gated (and why an operator scopes it with allow/deny).
"""
from __future__ import annotations

import fnmatch
import os
import shlex
import subprocess
from typing import Optional

# --------------------------------------------------------------------------- #
# Limits + constants
# --------------------------------------------------------------------------- #
MAX_OUTPUT_CHARS = 20_000                      # cap per stdout/stderr stream
DEFAULT_EXEC_TIMEOUT = 120                     # seconds

# Secrets never inherited into an exec'd child (CLAUDE.md credential hygiene).
_SECRET_ENV = ("VENICE_API_KEY", "VENICE_EMBED_API_KEY")

# Substrings that turn `/bin/sh -c` into more than a single simple command. When an
# allowlist is active we reject any of them so leading-token allowlisting can't be
# bypassed by e.g. `allowed && rm -rf ~` or `allowed | sh`.
_SHELL_META = (";", "|", "&", "<", ">", "`", "$", "(", ")", "\n")


# --------------------------------------------------------------------------- #
# Result helpers (shared JSON shape with `_code`)
# --------------------------------------------------------------------------- #
def _err(message: str) -> dict:
    return {"status": "error", "message": message}


def _ok(**kw) -> dict:
    return {"status": "ok", **kw}


def _confirm(message: str) -> dict:
    return {"status": "confirmation_required", "message": message}


# --------------------------------------------------------------------------- #
# JSON schema helpers + exec schemas (confirm/max_spend/output_dir absent)
# --------------------------------------------------------------------------- #
def _p(typ, desc=None):
    d = {"type": typ}
    if desc:
        d["description"] = desc
    return d


def _obj(props, required=None):
    s = {"type": "object", "properties": props}
    if required:
        s["required"] = required
    return s


_RUN_SCHEMA = _obj(
    {
        "command": _p("string", "Shell command to run (via /bin/sh -c) in the root."),
        "timeout": _p("integer", "Timeout in seconds for this command."),
    },
    ["command"],
)
_GIT_SCHEMA = _obj(
    {
        "subcommand": _p("string", "Read-only git subcommand (status/diff/log/show/...)."),
        "args": {"type": "array", "items": {"type": "string"},
                 "description": "Extra arguments for the subcommand."},
    },
    ["subcommand"],
)


# --------------------------------------------------------------------------- #
# Environment + policy
# --------------------------------------------------------------------------- #
def _scrubbed_env() -> dict:
    return {k: v for k, v in os.environ.items() if k not in _SECRET_ENV}


def _split(command: str):
    """shlex-split `command`, or None if it can't be parsed (unbalanced quotes)."""
    try:
        return shlex.split(command)
    except ValueError:
        return None


def check_policy(command, *, allow=(), deny=()) -> Optional[str]:
    """Return a human-readable refusal message if `command` is blocked, else None.

    Deny globs (on the full string + each token/basename) are always enforced and
    win over allow. A non-empty allowlist additionally requires a single simple
    command whose leading token's basename matches an allow entry. See the module
    docstring for the exact semantics.
    """
    cmd = str(command)
    tokens = _split(cmd)

    deny = [str(d) for d in (deny or [])]
    if deny:
        haystacks = [cmd]
        for tok in (tokens or []):
            haystacks.append(tok)
            haystacks.append(os.path.basename(tok))
        for pat in deny:
            if any(fnmatch.fnmatch(h, pat) for h in haystacks):
                return f"blocked by shell deny policy ({pat!r}): {cmd}"

    allow = [str(a) for a in (allow or [])]
    if allow:
        hit = next((m for m in _SHELL_META if m in cmd), None)
        if hit is not None:
            return (
                "shell allowlist is active, so only a single simple command is "
                "permitted (no operators/pipes/redirects/substitutions/variables; "
                f"found {hit!r}): {cmd}"
            )
        if not tokens:
            return f"could not parse command for the shell allowlist: {cmd}"
        argv0 = os.path.basename(tokens[0])
        if not any(fnmatch.fnmatch(argv0, a) for a in allow):
            return (
                f"{argv0!r} is not in the shell allowlist "
                f"({', '.join(allow)}): {cmd}"
            )
    return None


# --------------------------------------------------------------------------- #
# Exec primitives
# --------------------------------------------------------------------------- #
def run_cmd(root: str, command, *, timeout=None, exec_timeout: int = DEFAULT_EXEC_TIMEOUT,
            confirm: bool = False, allow=(), deny=()) -> dict:
    if not command or not str(command).strip():
        return _err("command is required")
    blocked = check_policy(command, allow=allow, deny=deny)
    if blocked:
        return _err(blocked)  # never confirmable -> refuse before the gate
    if not confirm:
        return _confirm(f"run will execute in {root}:\n    {command}")
    try:
        t = int(timeout) if timeout else int(exec_timeout)
    except (TypeError, ValueError):
        t = int(exec_timeout)
    try:
        proc = subprocess.run(
            ["/bin/sh", "-c", str(command)], cwd=root, capture_output=True,
            text=True, timeout=t, env=_scrubbed_env(),
        )
    except subprocess.TimeoutExpired:
        return _err(f"command timed out after {t}s")
    except OSError as e:
        return _err(f"could not run command: {e}")
    out, errout = proc.stdout or "", proc.stderr or ""
    return _ok(
        exit_code=proc.returncode,
        stdout=out[:MAX_OUTPUT_CHARS],
        stderr=errout[:MAX_OUTPUT_CHARS],
        truncated=(len(out) > MAX_OUTPUT_CHARS or len(errout) > MAX_OUTPUT_CHARS),
    )


_GIT_READONLY = frozenset({
    "status", "diff", "log", "show", "branch", "ls-files", "blame", "remote",
    "rev-parse", "describe", "shortlog",
})


def git_cmd(root: str, subcommand, *, args=None,
            exec_timeout: int = DEFAULT_EXEC_TIMEOUT) -> dict:
    sub = str(subcommand or "").strip()
    if sub not in _GIT_READONLY:
        return _err(
            "git: only read-only subcommands are allowed here "
            f"({', '.join(sorted(_GIT_READONLY))}); use the run tool "
            "(which confirms) for mutations like add/commit"
        )
    argv = ["git", sub]
    if args:
        if not isinstance(args, list):
            return _err("args must be a list of strings")
        for a in args:
            if not isinstance(a, (str, int, float)):
                return _err("each arg must be a string")
            argv.append(str(a))
    try:
        proc = subprocess.run(
            argv, cwd=root, capture_output=True, text=True,
            timeout=int(exec_timeout), env=_scrubbed_env(),
        )
    except FileNotFoundError:
        return _err("git is not installed")
    except subprocess.TimeoutExpired:
        return _err("git command timed out")
    except OSError as e:
        return _err(f"git failed: {e}")
    out, errout = proc.stdout or "", proc.stderr or ""
    return _ok(
        exit_code=proc.returncode,
        stdout=out[:MAX_OUTPUT_CHARS],
        stderr=errout[:MAX_OUTPUT_CHARS],
        truncated=(len(out) > MAX_OUTPUT_CHARS),
    )
