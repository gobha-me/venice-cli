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

The free Git boundary is different: :func:`git_cmd` accepts only exact read forms,
requires its caller to resolve every literal path through project authority, and
neutralizes Git environment/config features that can write or execute helpers.
Package-authored review plumbing uses the private :func:`_git_cmd_internal`; its
controlled ``diff --no-index`` form is never reachable from the model schema.
"""
from __future__ import annotations

import fnmatch
import os
import re
import shlex
import subprocess
from typing import Callable, List, Optional, Tuple

# --------------------------------------------------------------------------- #
# Limits + constants
# --------------------------------------------------------------------------- #
MAX_OUTPUT_CHARS = 20_000                      # cap per stdout/stderr stream
DEFAULT_EXEC_TIMEOUT = 120                     # seconds

# Secrets never inherited into an exec'd child (AGENTS.md credential hygiene).
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
        "subcommand": _p(
            "string",
            "Read-only Git operation. Arguments are validated per operation; "
            "mutations require the confirmed run tool.",
        ),
        "args": {"type": "array", "items": {"type": "string"},
                 "description": "Validated flags/revisions for the operation. "
                                "Put literal project paths after -- and do NOT "
                                "repeat the subcommand itself."},
    },
    ["subcommand"],
)


# --------------------------------------------------------------------------- #
# Environment + policy
# --------------------------------------------------------------------------- #
def _scrubbed_env() -> dict:
    return {k: v for k, v in os.environ.items() if k not in _SECRET_ENV}


def _git_env() -> dict:
    """Return an exec environment that cannot redirect Git to helpers/config."""
    env = {k: v for k, v in _scrubbed_env().items() if not k.startswith("GIT_")}
    env.update({
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_PAGER": "cat",
        "GIT_OPTIONAL_LOCKS": "0",
    })
    return env


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
    "rev-parse", "describe", "shortlog", "merge-base",
})

# Revisions deliberately exclude Git's ``REV:path`` object syntax and option-like
# forms. Common refs, hashes, ancestry suffixes, and one range remain available.
_REV_ATOM = r"(?:HEAD|[A-Za-z0-9][A-Za-z0-9._/@{}^~+/-]*)"
_REV_RE = re.compile(rf"{_REV_ATOM}(?:(?:\.\.|\.\.\.){_REV_ATOM})?\Z")
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")
_CONTEXT_RE = re.compile(
    r"(?:-U\d{1,3}|--unified=\d{1,3}|--inter-hunk-context=\d{1,3})\Z"
)
_MAX_COUNT_RE = re.compile(r"--max-count=(\d+)\Z")

_STATUS_FLAGS = frozenset({
    "-s", "--short", "-b", "--branch", "--show-stash", "--porcelain",
    "--porcelain=v1", "--porcelain=v2", "--long", "--ahead-behind",
    "--no-ahead-behind", "--renames", "--no-renames", "--ignored",
    "--ignored=traditional", "--ignored=matching", "--ignored=no",
    "--untracked-files", "--untracked-files=all", "--untracked-files=normal",
    "--untracked-files=no", "-uno", "-unormal", "-uall", "-z", "--null",
})
_DIFF_FLAGS = frozenset({
    "--cached", "--staged", "--merge-base", "--stat", "--numstat",
    "--shortstat", "--name-only", "--name-status", "--check", "--summary",
    "--patch", "-p", "--no-patch", "-s", "--raw", "--no-color",
    "--color=never", "--minimal", "--patience", "--histogram", "-w",
    "--ignore-all-space", "-b", "--ignore-space-change", "--ignore-space-at-eol",
    "--ignore-blank-lines", "-W", "--function-context", "--word-diff",
    "--word-diff=plain", "--word-diff=color", "--word-diff=porcelain",
    "--no-renames",
})
_LOG_FLAGS = frozenset({
    "--oneline", "--decorate", "--decorate=short", "--decorate=full",
    "--no-decorate", "--graph", "--all", "--branches", "--remotes", "--tags",
    "--first-parent", "--merges", "--no-merges", "--reverse", "--topo-order",
    "--date-order", "--author-date-order", "--no-color", "--color=never",
})
_SHOW_FLAGS = frozenset({
    "--oneline", "--stat", "--numstat", "--shortstat", "--name-only",
    "--name-status", "--summary", "--patch", "-p", "--no-patch", "-s",
    "--raw", "--no-color", "--color=never", "-w", "--ignore-all-space",
    "-b", "--ignore-space-change", "-W", "--function-context", "--no-renames",
})
_LS_FILES_FLAGS = frozenset({
    "--cached", "-c", "--deleted", "-d", "--modified", "-m", "--others", "-o",
    "--ignored", "-i", "--stage", "-s", "--unmerged", "-u", "--killed", "-k",
    "--directory", "--no-empty-directory", "--exclude-standard", "-z",
    "--deduplicate", "--sparse",
})
_BLAME_FLAGS = frozenset({
    "--line-porcelain", "--porcelain", "-w", "--show-stats", "--root",
    "--first-parent", "--show-name", "--show-number", "--score-debug",
})
_SHORTLOG_FLAGS = frozenset({"-n", "--numbered", "-s", "--summary", "-e", "--email"})
_DESCRIBE_FLAGS = frozenset({
    "--always", "--tags", "--all", "--long", "--exact-match", "--dirty",
    "--contains", "--first-parent",
})
_REV_PARSE_FLAGS = frozenset({
    "--verify", "--quiet", "-q", "--abbrev-ref", "--symbolic-full-name",
    "--short", "--show-toplevel", "--show-prefix", "--show-cdup",
    "--is-inside-work-tree", "--is-bare-repository", "--is-shallow-repository",
})
_MERGE_BASE_FLAGS = frozenset({
    "--all", "--octopus", "--independent", "--is-ancestor", "--fork-point",
})


class _GitPolicyError(ValueError):
    pass


def _safe_revision(value: str) -> bool:
    if value.startswith("-") or ":" in value or "\\" in value:
        return False
    return bool(_REV_RE.fullmatch(value))


def _split_paths(args: List[str]) -> Tuple[List[str], List[str]]:
    if args.count("--") > 1:
        raise _GitPolicyError("git: '--' may appear only once")
    if "--" not in args:
        return args, []
    at = args.index("--")
    paths = args[at + 1:]
    if not paths:
        raise _GitPolicyError("git: '--' must be followed by a project path")
    return args[:at], paths


def _bounded_max_count(token: str) -> bool:
    match = _MAX_COUNT_RE.fullmatch(token)
    return bool(match and 1 <= int(match.group(1)) <= 200)


def _option_with_value(token: str, prefixes) -> bool:
    return any(token.startswith(prefix) and len(token) > len(prefix) for prefix in prefixes)


def _validate_simple(before: List[str], flags, *, max_revs: int,
                     value_prefixes=()) -> None:
    revs = 0
    i = 0
    while i < len(before):
        token = before[i]
        if token in flags or _CONTEXT_RE.fullmatch(token):
            i += 1
            continue
        if _bounded_max_count(token):
            i += 1
            continue
        if token in ("-n", "--max-count"):
            if i + 1 >= len(before) or not before[i + 1].isdigit() \
                    or not 1 <= int(before[i + 1]) <= 200:
                raise _GitPolicyError("git: max-count must be an integer from 1 to 200")
            i += 2
            continue
        if _option_with_value(token, value_prefixes):
            i += 1
            continue
        if _safe_revision(token) and revs < max_revs:
            revs += 1
            i += 1
            continue
        raise _GitPolicyError(f"git: argument is not allowed for this read form: {token!r}")


def _validate_branch(before: List[str]) -> None:
    if not before:
        return
    listing = "--list" in before
    no_value = {
        "--list", "-a", "--all", "-r", "--remotes", "-v", "-vv", "--verbose",
        "--no-color", "--color=never", "--show-current", "--column", "--no-column",
    }
    value_prefixes = (
        "--contains=", "--no-contains=", "--merged=", "--no-merged=",
        "--points-at=", "--sort=", "--column=",
    )
    for token in before:
        if token in no_value:
            continue
        if _option_with_value(token, value_prefixes):
            value = token.split("=", 1)[1]
            if token.startswith((
                    "--contains=", "--no-contains=", "--merged=",
                    "--no-merged=", "--points-at=",
            )) and not _safe_revision(value):
                raise _GitPolicyError(f"git: unsafe branch revision: {value!r}")
            continue
        if listing and not token.startswith("-") and not _CONTROL_RE.search(token):
            continue
        raise _GitPolicyError(f"git: branch permits listing forms only: {token!r}")


def _validate_args(sub: str, args: List[str],
                   path_guard: Callable[[str], str]) -> List[str]:
    before, paths = _split_paths(args)
    if any(_CONTROL_RE.search(token) for token in args):
        raise _GitPolicyError("git: control characters are not allowed in arguments")

    if sub == "status":
        if any(token not in _STATUS_FLAGS for token in before):
            raise _GitPolicyError("git: status accepts display flags and '--' paths only")
    elif sub == "diff":
        _validate_simple(
            before, _DIFF_FLAGS, max_revs=2,
            value_prefixes=("--word-diff-regex=",),
        )
    elif sub == "log":
        _validate_simple(
            before, _LOG_FLAGS, max_revs=1,
            value_prefixes=(
                "--date=", "--author=", "--grep=", "--since=", "--until=",
                "--after=", "--before=",
            ),
        )
    elif sub == "show":
        _validate_simple(before, _SHOW_FLAGS, max_revs=1)
    elif sub == "ls-files":
        if any(token not in _LS_FILES_FLAGS for token in before):
            raise _GitPolicyError("git: ls-files option is not in the read allowlist")
    elif sub == "blame":
        i = 0
        revs = 0
        while i < len(before):
            token = before[i]
            if token in _BLAME_FLAGS:
                i += 1
            elif token == "-L" and i + 1 < len(before) \
                    and re.fullmatch(r"\d+(?:,\d+)?", before[i + 1]):
                i += 2
            elif token.startswith("-L") and re.fullmatch(r"-L\d+(?:,\d+)?", token):
                i += 1
            elif _safe_revision(token) and revs == 0:
                revs += 1
                i += 1
            else:
                raise _GitPolicyError(f"git: blame argument is not allowed: {token!r}")
        if len(paths) != 1:
            raise _GitPolicyError("git: blame requires exactly one project path after '--'")
    elif sub == "branch":
        if paths:
            raise _GitPolicyError("git: branch does not accept path operands")
        _validate_branch(before)
    elif sub == "remote":
        if before or paths:
            raise _GitPolicyError("git: remote permits names-only listing with no arguments")
    elif sub == "rev-parse":
        has_revision = False
        for token in before:
            if token in _REV_PARSE_FLAGS or re.fullmatch(r"--short=\d{1,2}", token):
                continue
            if _safe_revision(token):
                has_revision = True
                continue
            raise _GitPolicyError(f"git: rev-parse argument is not allowed: {token!r}")
        if has_revision and "--verify" not in before:
            raise _GitPolicyError("git: rev-parse revisions require --verify")
        if paths:
            raise _GitPolicyError("git: rev-parse does not accept path operands")
    elif sub == "describe":
        for token in before:
            if token in _DESCRIBE_FLAGS or re.fullmatch(r"--abbrev=\d{1,2}", token) \
                    or _option_with_value(token, ("--match=", "--exclude=", "--candidates=")):
                continue
            if _safe_revision(token):
                continue
            raise _GitPolicyError(f"git: describe argument is not allowed: {token!r}")
        if paths:
            raise _GitPolicyError("git: describe does not accept path operands")
    elif sub == "shortlog":
        _validate_simple(before, _SHORTLOG_FLAGS, max_revs=2)
        if paths:
            raise _GitPolicyError("git: shortlog does not accept path operands")
    elif sub == "merge-base":
        for token in before:
            if token in _MERGE_BASE_FLAGS or _safe_revision(token):
                continue
            raise _GitPolicyError(f"git: merge-base argument is not allowed: {token!r}")
        if paths:
            raise _GitPolicyError("git: merge-base does not accept path operands")
    else:
        raise _GitPolicyError(f"git: unsupported read operation: {sub!r}")

    guarded = []
    for path in paths:
        if path.startswith(":") or any(ch in path for ch in "*?["):
            raise _GitPolicyError(f"git: pathspec magic/globs are not allowed: {path!r}")
        try:
            guarded.append(path_guard(path))
        except Exception as exc:
            raise _GitPolicyError(str(exc)) from exc
    if sub == "diff" and not guarded:
        metadata = {
            "--stat", "--numstat", "--shortstat", "--name-only",
            "--name-status", "--raw", "--summary",
        }
        if not any(token in metadata for token in before):
            raise _GitPolicyError(
                "git: content diffs require one or more approved paths after '--'"
            )
    if sub == "show" and not guarded and not ({"--no-patch", "-s"} & set(before)):
        raise _GitPolicyError(
            "git: show without an approved path requires --no-patch"
        )
    if guarded:
        return before + ["--"] + guarded
    # Disambiguate revision-looking values from implicit filesystem operands.
    if sub in ("diff", "log", "show"):
        return before + ["--"]
    return before


def _run_git(root: str, sub: str, args: List[str], *, exec_timeout: int) -> dict:
    """Execute code-authored Git argv with helper execution neutralized."""
    argv = [
        "git", "--no-pager",
        "-c", "core.fsmonitor=false",
        "-c", f"core.hooksPath={os.devnull}",
        "-c", "diff.external=",
        "-c", "interactive.diffFilter=",
        sub,
    ]
    if sub in ("diff", "log", "show"):
        argv.extend(("--no-ext-diff", "--no-textconv"))
    argv.extend(args)
    try:
        proc = subprocess.run(
            argv, cwd=root, capture_output=True, text=True,
            timeout=int(exec_timeout), env=_git_env(),
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


def _git_cmd_internal(root: str, subcommand: str, args, *,
                      exec_timeout: int = DEFAULT_EXEC_TIMEOUT) -> dict:
    """Run package-authored read-only Git argv outside the model tool schema."""
    if subcommand not in _GIT_READONLY:
        return _err(f"git: internal read operation is not supported: {subcommand!r}")
    if not isinstance(args, list) or not all(isinstance(a, str) for a in args):
        return _err("git: internal args must be a list of strings")
    return _run_git(os.path.realpath(root), subcommand, args,
                    exec_timeout=exec_timeout)


def git_cmd(root: str, subcommand, *, args=None,
            path_guard: Optional[Callable[[str], str]] = None,
            exec_timeout: int = DEFAULT_EXEC_TIMEOUT) -> dict:
    """Run one strictly validated, argument-aware, read-only Git operation."""
    sub = str(subcommand or "").strip()
    if sub not in _GIT_READONLY:
        return _err(
            "git: only validated read-only operations are allowed here "
            f"({', '.join(sorted(_GIT_READONLY))}); use the run tool "
            "(which confirms) for mutations like add/commit"
        )
    if args is None:
        args = []
    if not isinstance(args, list) or not all(isinstance(a, str) for a in args):
        return _err("args must be a list of strings")
    if args and args[0].strip() == sub:
        return _err(
            f"git: don't repeat the subcommand in 'args'; pass only the "
            f"flags/arguments that follow {sub!r} (got args starting with "
            f"{sub!r}, which would run 'git {sub} {sub} ...')"
        )
    if path_guard is None:
        def path_guard(_path: str) -> str:
            raise _GitPolicyError("git: path authority is unavailable")
    try:
        safe_args = _validate_args(sub, args, path_guard)
    except (OSError, ValueError) as exc:
        return _err(str(exc))
    return _run_git(os.path.realpath(root), sub, safe_args,
                    exec_timeout=exec_timeout)
