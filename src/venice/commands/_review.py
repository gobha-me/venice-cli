"""Cold-context, diff-scoped code review (#80 part 1a).

The engine behind both entry points: the `venice review` subcommand (human-invoked)
and the `venice_review` rail tool (so a coding agent can review its own work mid-run
and fix findings before handoff). One engine, two callers -- `review.py` and
`code.py` differ only in how they present the result.

Why cold context. This is the venice-cli pilot for the quality pipeline in
aiforge#19. The evidence: cold-context review of *merged* termforge work found 10
confirmed bugs in v0.1.0 plus a shipped-unusable feature in v0.1.2 -- all with tests
green across 11 build configs. Independent eyes catch what config-matrix breadth
cannot, and precise findings (repro + `file:line`) produced a 6/6 fix rate. So the
reviewer runs on a FRESH context (`_agent.run_review` over `_run_disposable`) and,
where the catalog allows, on a DIFFERENT model than the author -- correlated blind
spots are the failure mode a same-model self-review cannot escape.

THE CONSTRAINT THIS MODULE IS BUILT AROUND
------------------------------------------
**Producing findings and certifying a diff are separate operations. This module only
ever does the first.**

It writes nothing: no receipt, no signature, no approval artifact, no session, not
one byte to disk. That is not an oversight to be fixed later -- it is the design. The
coding agent holds `apply_patch` and `shell`, so any receipt it could write it would
eventually write, not adversarially but because shortest-path-to-green is ordinary
agent behaviour. A gate inside the author's blast radius is not a gate. Fuse the two
operations here and #80 part 1b (real gating) becomes unwinnable.

Consequences that look like limitations and are not:
- The exit code reports what one run found. It is NOT certification. An authoring
  agent may ignore it or pass `--fail-on none`; that is expected, because the code is
  for a human's shell and for part 1b's CI job, where the agent is not the invoker.
- `status` distinguishes "skipped" from "clean". Collapsing them would let "we never
  looked" masquerade as "we looked and found nothing" -- the same honour-system
  failure in miniature.
- `base_sha`/`head_sha` are reported on every path. Part 1b needs exactly that pair to
  bind a future check to a specific diff; producing it costs nothing now.

Cost discipline (aiforge#19 SS5). The reviewer reads a diff, never the repo: the
manifest is fetched once, files are classified, and a docs/test-only change skips the
model entirely -- before the SDK is imported or a client is built, so it costs zero
API calls. Enclosing-function context comes from `git diff -W`, which git implements
per-language via its userdiff drivers; a hand-rolled extractor would have to be
`ast`-based to stay stdlib-only, i.e. Python-only, which would do nothing for the
C++23 repo this is aimed at.

Import discipline: stdlib only (mirrors `_agent`/`_code`). The `openai` SDK is the
caller's business.
"""
from __future__ import annotations

import os
import posixpath
import re
import threading
import time
from typing import Dict, List, Optional, Tuple

from . import _agent, _code, _exec, _openai
from ._exec import (  # shared exec rails (#33): one gate for every git shell-out
    DEFAULT_EXEC_TIMEOUT,
    MAX_OUTPUT_CHARS,
    _err,
    _obj,
    _p,
)

# --------------------------------------------------------------------------- #
# Limits
# --------------------------------------------------------------------------- #
#: Total characters of diff handed to the reviewer. ~15k tokens, which sits
#: comfortably alongside REVIEW_SYSTEM. Per-file output is separately capped by
#: `MAX_OUTPUT_CHARS`, so there is exactly one truncation boundary per file
#: and one for the whole payload -- never two that can disagree.
MAX_DIFF_CHARS = 60_000

#: Diff context lines. Rides on top of `-W`, which expands to function boundaries;
#: -U8 is what shows up where `-W` finds no function (headers, config, markup).
DIFF_CONTEXT_LINES = 8

REVIEW_MAX_TOOL_CALLS = 8        # default reads-beyond-the-diff budget per round
REVIEW_HARD_CAP = 20             # ceiling regardless of what a caller asks for
REVIEW_DEFAULT_ROUNDS = 2        # `venice review`: one pass, then one look for misses
REVIEW_TOOL_ROUNDS = 1           # the rail: the agent's own fix-loop is the outer loop
REVIEW_HARD_ROUNDS = 3           # aiforge#19 SS5: until-dry, capped at 2-3

#: Reviews one `venice code` session may run. Structural, not prompt-enforced: a
#: fix-review-fix spiral is the obvious failure mode and prose would not stop it.
REVIEW_MAX_INVOCATIONS = 3

#: Ordered so `--fail-on` can compare. A finding whose severity is missing or
#: unrecognised is treated as `major` -- see `_severity`.
SEVERITY_ORDER = {"minor": 1, "major": 2, "blocker": 3}
FAIL_ON_CHOICES = ("none", "minor", "major", "blocker")
DEFAULT_FAIL_ON = "major"
CONTEXT_CHOICES = ("function", "hunk")
EFFORT_CHOICES = ("auto", "always", "never")


# --------------------------------------------------------------------------- #
# Surface triage
# --------------------------------------------------------------------------- #
_GENERATED_NAMES = frozenset({
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml", "poetry.lock",
    "cargo.lock", "go.sum", "composer.lock", "gemfile.lock", "uv.lock",
})
_GENERATED_SUFFIXES = (".min.js", ".min.css", ".map", ".lock")
_TEST_DIRS = frozenset({"tests", "test", "spec", "specs", "__tests__",
                        "testdata", "fixtures"})
_TEST_PATTERNS = (
    re.compile(r"^test_.*\.py$"), re.compile(r"^.*_test\.py$"),
    re.compile(r"^.*_test\.(go|cc|cpp|cxx|rs|c)$"),
    re.compile(r"^.*[Tt]est\.(java|kt|cs)$"),
    re.compile(r"^.*\.(test|spec)\.[jt]sx?$"),
    re.compile(r"^conftest\.py$"),
)
_DOC_SUFFIXES = (".md", ".markdown", ".rst", ".txt", ".adoc")
_DOC_STEMS = frozenset({
    "readme", "license", "licence", "changelog", "contributing",
    "code_of_conduct", "security", "notice", "authors", "copying",
})


def classify(path: str, added: str = "", deleted: str = "") -> str:
    """Bucket one changed path: binary | generated | docs | test | code.

    First match wins. Deliberately SHORT -- over-listing is how real code gets
    silently skipped, and a skipped review that reports success is the worst
    failure this module can have. When in doubt a path falls through to `code`,
    which costs a review rather than missing one.
    """
    if added == "-" and deleted == "-":     # numstat's binary marker
        return "binary"
    rel = (path or "").replace("\\", "/").lstrip("./").lower()
    if not rel:
        return "code"
    base = posixpath.basename(rel)
    parts = rel.split("/")
    stem, _, ext = base.rpartition(".")
    if base in _GENERATED_NAMES or rel.endswith(_GENERATED_SUFFIXES):
        return "generated"
    if any(p in _TEST_DIRS for p in parts[:-1]) or any(
            p.match(base) for p in _TEST_PATTERNS):
        return "test"
    if rel.endswith(_DOC_SUFFIXES) or parts[0] in ("docs", "doc") or (
            (stem or base) in _DOC_STEMS):
        return "docs"
    return "code"


# --------------------------------------------------------------------------- #
# git plumbing (code-authored argv use `_exec`'s private hardened runner)
# --------------------------------------------------------------------------- #
class ReviewError(Exception):
    """A review could not be set up. `.message` is printable; `.code` is the exit code."""

    def __init__(self, message: str, code: int = 2):
        super().__init__(message)
        self.message = message
        self.code = code


def _git(root: str, sub: str, args, exec_timeout: int,
         ok_codes: Tuple[int, ...] = (0,)) -> Tuple[bool, str]:
    """Run one read-only git subcommand; return (ok, stripped stdout).

    ``ok_codes`` exists for ``git diff --no-index``, which reports "the files
    differ" as exit 1 -- the normal, expected outcome when diffing an untracked
    file against /dev/null.
    """
    out = _exec._git_cmd_internal(
        root, sub, args=list(args), exec_timeout=exec_timeout
    )
    if out.get("status") != "ok" or out.get("exit_code") not in ok_codes:
        return False, (out.get("stderr") or out.get("message") or "").strip()
    return True, (out.get("stdout") or "").strip()


#: Probed in order when no base is given. `origin/HEAD` is checked first but is
#: genuinely often absent -- a plain `git clone` sets it, but repos created locally
#: and pushed (this one included) have no such ref -- so the candidate probe is the
#: primary path, not a fallback for an edge case.
_BASE_CANDIDATES = ("origin/main", "origin/master", "main", "master")


def resolve_base(root: str, requested: Optional[str], exec_timeout: int) -> Tuple[str, str]:
    """Resolve the review base to (ref, note). Raises ReviewError if unusable."""
    ok, _ = _git(root, "rev-parse", ["--git-dir"], exec_timeout)
    if not ok:
        raise ReviewError(f"not a git repository (or git is unavailable): {root}")
    ok, _ = _git(root, "rev-parse", ["--verify", "--quiet", "HEAD"], exec_timeout)
    if not ok:
        raise ReviewError("this repository has no commits yet; nothing to review")

    if requested:
        ok, _ = _git(root, "rev-parse", ["--verify", "--quiet", requested], exec_timeout)
        if not ok:
            raise ReviewError(f"unknown --base ref {requested!r}")
        return requested, ""

    ok, head = _git(root, "rev-parse", ["--abbrev-ref", "origin/HEAD"], exec_timeout)
    if ok and head and head != "origin/HEAD":
        return head, f"base auto-detected from origin/HEAD: {head}"
    for cand in _BASE_CANDIDATES:
        ok, _ = _git(root, "rev-parse", ["--verify", "--quiet", cand], exec_timeout)
        if ok:
            return cand, f"base auto-detected: {cand}"
    return "HEAD", ("no default branch found (no origin/HEAD, main or master); "
                    "reviewing uncommitted changes only")


def _parse_numstat_z(out: str) -> List[Tuple[str, str, str]]:
    """Parse `git diff --numstat -z` into (added, deleted, path) triples.

    With -z a normal record is ``added\\tdeleted\\tpathNUL``; a rename/copy is
    ``added\\tdeleted\\tNUL`` followed by two more NUL records (old path, new path).
    We keep the NEW path -- that is what a reviewer reads. Renames are left ON
    (`--no-renames` would turn one move into two whole-file diffs, which is exactly
    the cost blowup diff-scoping exists to avoid).
    """
    parts = out.split("\0")
    entries: List[Tuple[str, str, str]] = []
    i = 0
    while i < len(parts):
        rec = parts[i]
        if not rec.strip():
            i += 1
            continue
        fields = rec.split("\t")
        if len(fields) < 3:
            i += 1
            continue
        added, deleted, path = fields[0], fields[1], fields[2]
        if path == "":                      # rename/copy: old, new follow
            if i + 2 < len(parts):
                path = parts[i + 2]
                i += 3
            else:
                i += 1
                continue
        else:
            i += 1
        if path:
            entries.append((added, deleted, path))
    return entries


def collect_diff(root: str, base: str, *, paths=None, context: str = "function",
                 max_chars: int = MAX_DIFF_CHARS,
                 exec_timeout: int = DEFAULT_EXEC_TIMEOUT) -> dict:
    """Collect the diff to review, scoped and budgeted.

    Returns a dict with the payload text plus the provenance and skip bookkeeping
    the callers surface. Raises :class:`ReviewError` when the repo is unusable.

    The range is ``merge-base(base, HEAD)`` resolved to a SHA **once**, then pinned:
    a single-commit-arg `git diff` compares that commit against the WORKING TREE, so
    one default covers both "review my branch before the PR" and an agent's "review
    what I just wrote". Pinning to the SHA (rather than re-deriving per call) means a
    moving `origin/master` cannot silently shift the range between rounds, and gives
    the (base_sha, head_sha) pair part 1b needs.
    """
    ok, base_sha = _git(root, "merge-base", [base, "HEAD"], exec_timeout)
    if not ok or not base_sha:
        # Unrelated histories, or a base that shares no ancestor: fall back to a
        # direct comparison rather than failing the review outright.
        base_sha = base
    ok, head_sha = _git(root, "rev-parse", ["HEAD"], exec_timeout)
    if not ok:
        raise ReviewError("could not resolve HEAD")

    pathspec = ["--"] + [str(p) for p in paths] if paths else []
    ok, manifest = _git(root, "diff", ["--numstat", "-z", base_sha] + pathspec,
                        exec_timeout)
    if not ok:
        raise ReviewError(f"git diff failed: {manifest or 'unknown error'}")

    entries = _parse_numstat_z(manifest)

    # Untracked files. `git diff` cannot see them at all, but a file the agent just
    # CREATED and has not staged is precisely the code most in need of review -- and
    # a reviewer that silently ignores whole new files while reporting success is the
    # worst failure this module could have. Each is diffed against /dev/null, which
    # is also how git renders a genuine new-file hunk, so the payload looks identical
    # whether or not the author remembered to `git add`.
    untracked: List[str] = []
    ok, listing = _git(root, "ls-files",
                       ["--others", "--exclude-standard", "-z"] + pathspec,
                       exec_timeout)
    if ok:
        for rel in listing.split("\0"):
            if not rel.strip():
                continue
            got, stat = _git(root, "diff",
                             ["--no-index", "--numstat", "-z", "--", os.devnull, rel],
                             exec_timeout, ok_codes=(0, 1))
            if not got:
                continue
            for added, deleted, _p in _parse_numstat_z(stat):
                entries.append((added, deleted, rel))
                untracked.append(rel)
                break
    # Sorted so payload order depends on the paths, not on the order git happened to
    # list them (tracked come from `diff`, untracked from `ls-files`) -- the same
    # not-positionally-biased principle as the budget selection below.
    entries.sort(key=lambda e: e[2])
    counts: Dict[str, int] = {}
    reviewable: List[Tuple[str, str, str]] = []
    skipped: List[dict] = []
    for added, deleted, path in entries:
        kind = classify(path, added, deleted)
        counts[kind] = counts.get(kind, 0) + 1
        if kind == "code":
            reviewable.append((added, deleted, path))
        else:
            skipped.append({"path": path, "reason": kind})

    ctx = ["-W"] if context == "function" else []
    untracked_set = set(untracked)
    truncated_files: List[str] = []
    collected: List[Tuple[str, str]] = []       # (path, diff text), in path order
    for _added, _deleted, path in reviewable:
        def _diff(with_ctx: bool, path=path):
            head = (ctx if with_ctx else []) + [f"-U{DIFF_CONTEXT_LINES}"]
            if path in untracked_set:   # exit 1 == "they differ", the normal case
                return _git(root, "diff", ["--no-index"] + head +
                            ["--", os.devnull, path], exec_timeout, ok_codes=(0, 1))
            return _git(root, "diff", head + [base_sha, "--", path], exec_timeout)

        got, text = _diff(True)
        if not got and ctx:
            # `-W` needs a userdiff driver for the language; on any failure retry
            # without it rather than dropping the file from the review entirely.
            got, text = _diff(False)
        if not got or not text:
            continue
        if len(text) >= MAX_OUTPUT_CHARS:
            truncated_files.append(path)
        collected.append((path, text))

    # Budget selection. Deliberately NOT first-come-first-served over the manifest:
    # untracked files are discovered after the tracked ones, so a positional rule
    # dropped brand-new files first -- and in an agent workflow a file that was just
    # CREATED is the most important thing in the diff. Observed live: a real feature
    # branch reviewed six edited files and omitted both new modules.
    #
    # Smallest-first instead, which maximises the number of files covered and makes
    # the choice depend on the diff rather than on git's listing order. Whatever does
    # not fit is NAMED (never silently clipped), and the reviewer holds `read_file`.
    total = sum(len(t) for _p, t in collected)
    keep = {p for p, _t in collected}
    if total > max_chars:
        keep, running = set(), 0
        for path, text in sorted(collected, key=lambda pt: len(pt[1])):
            if running + len(text) > max_chars and keep:
                continue
            keep.add(path)
            running += len(text)
    chunks = [t for p, t in collected if p in keep]
    reviewed = [p for p, _t in collected if p in keep]
    omitted = [p for p, _t in collected if p not in keep]

    return {
        "base": base,
        "base_sha": base_sha,
        "head_sha": head_sha,
        "diff": "\n".join(chunks),
        "files_reviewed": reviewed,
        "files_skipped": skipped,
        "files_omitted": omitted,
        "files_truncated": truncated_files,
        "counts": counts,
        "diff_truncated": bool(omitted or truncated_files),
        "changed_paths": [p for _a, _d, p in entries],
    }


# --------------------------------------------------------------------------- #
# The task the reviewer is handed
# --------------------------------------------------------------------------- #
def build_task(collected: dict, *, prior=None) -> str:
    """Assemble the reviewer's user turn: provenance, caveats, then the diff.

    Prior findings ride HERE after the closing diff fence, not in the system prompt or
    ahead of the diff. ``REVIEW_SYSTEM`` and the entire first-round task therefore stay
    a byte-identical prefix of later rounds (what keeps the input cache hot --
    aiforge#19 SS5 -- and keeps the prompt pins meaningful).
    """
    head = [
        f"Review the following diff. Base: {collected['base']} "
        f"({collected['base_sha'][:12]}), HEAD {collected['head_sha'][:12]}.",
        f"Files in scope ({len(collected['files_reviewed'])}): "
        + ", ".join(collected["files_reviewed"]),
    ]
    if collected["files_truncated"]:
        head.append(
            "TRUNCATED (too large to include whole -- read them with read_file if "
            "you need more): " + ", ".join(collected["files_truncated"]))
    if collected["files_omitted"]:
        head.append(
            "NOT INCLUDED (diff budget exhausted -- your review does not cover "
            "these; say so under NOT CHECKED): " + ", ".join(collected["files_omitted"]))
    if collected["files_skipped"]:
        head.append(
            "Skipped as non-code: "
            + ", ".join(f"{s['path']} ({s['reason']})" for s in collected["files_skipped"]))
    task = "\n".join(head) + "\n\n```diff\n" + collected["diff"] + "\n```\n"
    if prior:
        task += (
            "\nALREADY REPORTED in an earlier pass over this same diff -- do NOT "
            "repeat these; look for what they missed:\n"
            + "\n".join(f"  - {f['file']}:{f['line']} {f['summary']}" for f in prior)
            + "\n"
        )
    return task


# --------------------------------------------------------------------------- #
# Verdict + findings parsing
# --------------------------------------------------------------------------- #
#: Mirrors `code._VERDICT_RE` (#25): loose on purpose -- `findall` + last-match-wins,
#: case-insensitive, unanchored, so `**review: clean**` in markdown still parses.
VERDICT_RE = re.compile(r"REVIEW:\s*(CLEAN|FINDINGS)", re.IGNORECASE)

RETRY_MSG = (
    "Your reply did not end with the required verdict line. Reply with nothing but a "
    "single line that is exactly 'REVIEW: CLEAN' if you found no defects, or exactly "
    "'REVIEW: FINDINGS' if you listed one or more."
)

_FINDING_RE = re.compile(
    r"^\s*(?:[-*+]\s+)?(?P<file>[^\s:][^:]*?):(?P<line>\d+)(?:-\d+)?\s*"
    r"(?:[\[(](?P<severity>blocker|major|minor)[\])])?\s*[-:—]?\s*"
    r"(?P<summary>\S.*?)\s*$",
    re.IGNORECASE,
)
#: Continuation lines of a finding block. Never start a new finding, even though a
#: REPRO line can easily contain a `path.py:12`-shaped substring.
_CONT_RE = re.compile(r"^\s*(?:[-*+]\s+)?(?:WHY|REPRO|FIX)\b\s*:", re.IGNORECASE)


def parse_verdict(report: Optional[str]) -> Optional[str]:
    """'clean' / 'findings' from the last REVIEW sentinel, or None if absent."""
    m = VERDICT_RE.findall(report or "")
    return m[-1].lower() if m else None


def _severity(raw: Optional[str]) -> str:
    """Normalise a severity, failing CLOSED.

    A missing or unrecognised severity becomes `major`, never `minor`. Failing open
    would let sloppy formatting silently demote a blocker past `--fail-on` -- a
    quiet downgrade is worse than a noisy over-report.
    """
    val = (raw or "").strip().lower()
    return val if val in SEVERITY_ORDER else "major"


def _norm(path: str) -> str:
    return posixpath.normpath((path or "").replace("\\", "/").lstrip("./")) or "."


def _in_diff(path: str, changed: List[str]) -> bool:
    """True when a finding's path names a file the diff actually touched.

    Component-aligned suffix matching either way, so a reviewer that writes an
    absolute path or one relative to a subdirectory still lands in `findings`.
    """
    p = _norm(path)
    for c in changed:
        cn = _norm(c)
        if p == cn or p.endswith("/" + cn) or cn.endswith("/" + p):
            return True
    return False


def parse_findings(report: str, changed_paths=None) -> Tuple[List[dict], List[dict]]:
    """Extract structured findings from a report's FINDINGS section.

    Returns ``(findings, findings_outside_diff)``. Extraction is purely additive --
    the full prose is always kept by the caller in `report`, so a finding this misses
    is still visible to a human; nothing is ever lost by a parse failure.

    A finding naming a file the diff did not touch is separated rather than dropped
    (the reviewer may well be right about a caller it read), but it must not drive
    the exit code, or the scoping contract means nothing.
    """
    changed = list(changed_paths or [])
    fields = _agent._parse_sections(report or "", _agent.REVIEW_SECTIONS)
    body = fields.get("FINDINGS", "")
    inside: List[dict] = []
    outside: List[dict] = []
    current: Optional[dict] = None
    bucket: Optional[List[dict]] = None

    def _flush() -> None:
        if current is not None and bucket is not None:
            current["block"] = "\n".join(current["_lines"]).strip()
            del current["_lines"]
            bucket.append(current)

    for line in body.splitlines():
        if _CONT_RE.match(line):
            if current is not None:
                current["_lines"].append(line.strip())
            continue
        m = _FINDING_RE.match(line)
        if m and m.group("summary"):
            _flush()
            current = {
                "file": m.group("file").strip(),
                "line": int(m.group("line")),
                "severity": _severity(m.group("severity")),
                "summary": m.group("summary").strip(),
                "_lines": [line.strip()],
            }
            bucket = inside if _in_diff(current["file"], changed) else outside
        elif current is not None and line.strip():
            current["_lines"].append(line.strip())
    _flush()
    return inside, outside


def fails(findings: List[dict], fail_on: str) -> bool:
    """Whether these findings should make the caller exit non-zero."""
    if fail_on == "none":
        return False
    floor = SEVERITY_ORDER.get(fail_on, SEVERITY_ORDER[DEFAULT_FAIL_ON])
    return any(SEVERITY_ORDER.get(f["severity"], 2) >= floor for f in findings)


# --------------------------------------------------------------------------- #
# Reviewer model selection (decorrelation)
# --------------------------------------------------------------------------- #
def _family(model_id: str) -> str:
    """The leading vendor/family token of a model id.

    `qwen3-4b` and `qwen-2.5-coder` share family `qwen`, so picking a different *id*
    is not by itself decorrelation -- same family, largely the same blind spots.
    """
    # `maxsplit=` by keyword: 3.13 deprecates passing it positionally, and the
    # keyword form has been valid since long before the 3.9 floor.
    return re.split(r"[-._0-9]", (model_id or "").strip().lower(), maxsplit=1)[0]


def resolve_reviewer_model(models, requested: Optional[str],
                           author_model: str) -> Tuple[Optional[str], bool]:
    """Pick the reviewer's model. Returns (model_id, decorrelated).

    Shaped like `_agent.resolve_web_search_model` (#77): an explicit operator override
    is trusted as-is (the caller validates it through `_models.resolve_model`, so an
    unknown id still exits 6); otherwise prefer a function-calling model from a
    DIFFERENT family, then merely a different id, then give up and reuse the author's.

    Reusing the author's model is never an error. aiforge#19 says *prefer* a different
    model, and hard-failing would make `venice code --review` unusable on a
    single-model deployment. The caller warns loudly and reports `decorrelated: false`;
    whether a correlated review is good enough to GATE on is a policy question that
    belongs to part 1b, not to the finder.
    """
    if requested:
        return requested, _family(requested) != _family(author_model)
    ids = [m.get("id") for m in (models or [])
           if isinstance(m, dict) and m.get("id")]
    capable = [i for i in ids
               if _agent.supports_function_calling(models, i) is not False]
    for mid in capable:
        if _family(mid) != _family(author_model):
            return mid, True
    for mid in capable:
        if mid != author_model:
            return mid, False
    return author_model, False


# --------------------------------------------------------------------------- #
# The review cycle
# --------------------------------------------------------------------------- #
def _retry_for_verdict(oai, model: str, report: str, base_kwargs: dict,
                       *, ledger=None) -> str:
    """One extra completion asking only for the sentinel line.

    Mirrors `code.py`'s single re-prompt, but as a FRESH one-shot rather than a
    continuation, so `_agent._run_disposable` stays byte-identical (the
    behavior-preserving rule the #52 arc is built on).

    Records onto the cycle ledger (#81): this call runs outside `_run_disposable`,
    so without it the `tokens` figure the reviewer already advertises under-reports
    every run that needed a verdict re-prompt -- and it also escapes the token cap
    it is supposed to be bounded by.
    """
    _t0 = time.monotonic()
    # #128: this retry is intentionally a fresh one-shot, not a continuation of the
    # author or reviewer conversation, so it must not borrow either affinity identity.
    retry_kwargs = _openai.without_prompt_cache_key(base_kwargs)
    resp = oai.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _agent.REVIEW_SYSTEM},
            {"role": "assistant", "content": report or ""},
            {"role": "user", "content": RETRY_MSG},
        ],
        **retry_kwargs,
    )
    if ledger is not None:
        ledger.record(getattr(resp, "usage", None), seconds=time.monotonic() - _t0)
    if getattr(resp, "choices", None):
        return resp.choices[0].message.content or ""
    return ""


def _key(f: dict) -> tuple:
    return (_norm(f["file"]), f["line"], f["summary"][:60].lower())


def run_cycle(oai, model: str, collected: dict, base_kwargs: dict, *,
              root: str, client=None, rounds: int = REVIEW_DEFAULT_ROUNDS,
              max_tool_calls: int = REVIEW_MAX_TOOL_CALLS,
              max_tokens: Optional[int] = None, focus: Optional[str] = None,
              include_search: bool = False,
              models=None, parent_ledger=None) -> dict:
    """Run the until-dry review loop over an already-collected diff.

    Rounds are re-passes over the SAME diff: each one is told what earlier passes
    already reported and asked for what they missed. Stops early when a round adds
    nothing new ("dry") -- a plain count would keep paying for rounds that have
    stopped finding anything, and aiforge#19 caps the loop at 2-3 for exactly that
    reason.

    ONE ledger spans the whole cycle, so `--subagent-max-tokens` bounds the review
    rather than each round of it; a cycle that crosses the ceiling stops looping
    instead of buying another round it cannot afford. (If rounds ever get their own
    per-round ledgers, each needs its own mirror -- see below.)

    `models`/`parent_ledger` (#117): the ledger is PRICED against `model` -- the
    REVIEWER's model, which `--review-model` makes deliberately different from the
    author's, so pricing it anywhere but here would bill the reviewer's tokens at the
    author's rate -- and mirrored into the parent's `review` bucket. Both default to
    None because `venice review` (the standalone CLI) has no parent ledger and must
    keep working unchanged; there the cycle simply meters itself as before.

    The mirror lives on the ledger rather than at the call sites, which matters here
    more than on any other rail: this cycle makes an API call OUTSIDE `run_loop` (the
    verdict re-prompt), and a `run_loop`-level callback would have silently missed it.
    """
    rounds = max(1, min(int(rounds or 1), REVIEW_HARD_ROUNDS))
    calls = max(1, min(int(max_tool_calls or REVIEW_MAX_TOOL_CALLS), REVIEW_HARD_CAP))
    inner = _code.read_only_tools(root, client, include_search=include_search)
    ledger = _agent.subagent_ledger(
        models, model, max_tokens=max_tokens,
        mirror=(parent_ledger, "review") if parent_ledger is not None else None,
    )

    findings: List[dict] = []
    outside: List[dict] = []
    seen = set()
    reports: List[str] = []
    verdict: Optional[str] = None
    used = 0
    tool_calls = 0

    for n in range(rounds):
        task = build_task(collected, prior=findings if n else None)
        out = _agent.run_review(oai, model, task, inner, base_kwargs,
                                max_tool_calls=calls, ledger=ledger, focus=focus)
        used = n + 1
        if out.get("status") != "ok":
            if not reports:
                return {**out, "rounds": used,
                        "tokens": ledger.prompt_tokens + ledger.completion_tokens}
            break
        report = out.get("report") or ""
        reports.append(report)
        tool_calls += int(out.get("tool_calls") or 0)
        v = parse_verdict(report)
        if v is None:
            report = (f"{report}\n"
                      f"{_retry_for_verdict(oai, model, report, base_kwargs, ledger=ledger)}")
            reports[-1] = report
            v = parse_verdict(report)
        # Keep the last PARSEABLE verdict. A later round that forgets the sentinel
        # must not erase an earlier good one -- doing so turned a review that had
        # already found a blocker into exit 10, throwing away real findings.
        if v is not None:
            verdict = v
        elif verdict is None:
            verdict = "unknown"
        got_in, got_out = parse_findings(report, collected["changed_paths"])
        fresh = [f for f in got_in if _key(f) not in seen]
        for f in fresh:
            seen.add(_key(f))
        findings.extend(fresh)
        outside.extend(f for f in got_out if _key(f) not in seen)
        for f in got_out:
            seen.add(_key(f))
        if v is None:
            break                      # a report we can't read is not worth re-running
        if n and not fresh:
            break                      # dry
        if n == 0 and verdict == "clean" and not fresh:
            break                      # nothing found first time; a re-look is waste
        if ledger.over_tokens():
            break

    # Content beats sentinel. A model that lists defects and then types
    # 'REVIEW: CLEAN' is contradicting itself; trusting the findings fails closed.
    if findings and verdict in ("clean", None):
        verdict = "findings"
    return {
        "status": "ok",
        "verdict": verdict or "unknown",
        "findings": findings,
        "findings_outside_diff": outside,
        "report": "\n\n".join(reports).strip(),
        "rounds": used,
        "tool_calls": tool_calls,
        "tokens": ledger.prompt_tokens + ledger.completion_tokens,
        "token_cap": ledger.max_tokens,
    }


def envelope(collected: dict, result: dict, *, model: str,
             decorrelated: bool, fail_on: str) -> dict:
    """The shared result shape for both entry points.

    Deliberately carries NO approval/receipt/certification key at any depth, and a
    test pins that by regex -- so part 1b's tempting first draft ("just add an
    `approved` flag") fails loudly instead of quietly fusing findings with
    certification. `status` separates "skipped" from "clean" for the same reason.
    """
    findings = result.get("findings", [])
    return {
        "status": result.get("status", "ok"),
        "verdict": result.get("verdict"),
        "findings": findings,
        "findings_outside_diff": result.get("findings_outside_diff", []),
        "report": result.get("report", ""),
        "rounds": result.get("rounds", 0),
        "base": collected["base"],
        "base_sha": collected["base_sha"],
        "head_sha": collected["head_sha"],
        "files_reviewed": collected["files_reviewed"],
        "files_skipped": collected["files_skipped"],
        "files_omitted": collected["files_omitted"],
        "diff_truncated": collected["diff_truncated"],
        "model": model,
        "decorrelated": decorrelated,
        "fail_on": fail_on,
        "failed": fails(findings, fail_on),
        "tokens": result.get("tokens", 0),
        "token_cap": result.get("token_cap"),
        "tool_calls": result.get("tool_calls", 0),
    }


def skipped_envelope(collected: dict, *, reason: str, fail_on: str) -> dict:
    """The zero-cost result for a diff that never reached the model."""
    return {
        "status": "skipped",
        "verdict": None,
        "reason": reason,
        "findings": [],
        "findings_outside_diff": [],
        "report": "",
        "rounds": 0,
        "base": collected["base"],
        "base_sha": collected["base_sha"],
        "head_sha": collected["head_sha"],
        "files_reviewed": collected["files_reviewed"],
        "files_skipped": collected["files_skipped"],
        "files_omitted": collected["files_omitted"],
        "diff_truncated": collected["diff_truncated"],
        "model": None,
        "decorrelated": None,
        "fail_on": fail_on,
        "failed": False,
        "tokens": 0,
        "token_cap": None,
        "tool_calls": 0,
    }


def should_skip(collected: dict, effort: str) -> Optional[str]:
    """Reason to skip the model step, or None to review.

    Called BEFORE the SDK is imported or a client is built, so a docs-only diff costs
    zero API calls and does not even require the `[openai]` extra.

    Known hole, stated rather than hidden: a test-only diff that deletes assertions is
    real risk, and `auto` skips it. `--effort always` is the closer.
    """
    if effort == "never":
        return "review skipped (--effort never)"
    if not collected["changed_paths"]:
        return "empty diff -- nothing changed against the base"
    if effort == "always":
        return None
    if not collected["files_reviewed"]:
        counts = collected["counts"]
        detail = ", ".join(f"{v} {k}" for k, v in sorted(counts.items())) or "none"
        return f"no code files in the diff ({detail})"
    return None


# --------------------------------------------------------------------------- #
# The `venice_review` rail tool (#80 part 1a)
# --------------------------------------------------------------------------- #
_REVIEW_SCHEMA = _obj({
    "base": _p("string",
                     "Git ref to review against (default: the repository's default "
                     "branch, auto-detected). Pass 'HEAD' to review only uncommitted "
                     "edits."),
    "paths": {"type": "array", "items": {"type": "string"},
              "description": "Optional pathspecs narrowing the diff to these files."},
    "focus": _p("string",
                      "Optional hint: what to weigh most (a subsystem, a risk you "
                      "want checked). Not a hard scope."),
    "rounds": {"type": "integer", "minimum": 1, "maximum": REVIEW_HARD_ROUNDS,
               "description": f"Passes over the same diff (default "
                              f"{REVIEW_TOOL_ROUNDS}, max {REVIEW_HARD_ROUNDS}); each "
                              f"is a full model run, so raise it only when a first "
                              f"pass looked shallow."},
    "max_tool_calls": {"type": "integer", "minimum": 1, "maximum": REVIEW_HARD_CAP,
                       "description": f"Cap on the reviewer's own tool calls "
                                      f"(default {REVIEW_MAX_TOOL_CALLS}, max "
                                      f"{REVIEW_HARD_CAP})."},
})
# NOTE: `model` is deliberately absent -- the reviewer's model is operator-controlled
# and resolved once at factory time (mirroring `_code.web_search_tool`), so the agent
# cannot escalate itself onto a costlier model. Also absent, and far more important:
# anything resembling certify/approve/receipt/sign. The agent cannot even ASK to have
# its work certified, because that operation does not exist on this side of the line.


def review_tool(oai, model: str, root: str, client, base_kwargs, *,
                include_search: bool = False,
                default_rounds: int = REVIEW_TOOL_ROUNDS,
                default_max_tool_calls: int = REVIEW_MAX_TOOL_CALLS,
                max_invocations: int = REVIEW_MAX_INVOCATIONS,
                max_tokens: Optional[int] = None,
                exec_timeout: int = DEFAULT_EXEC_TIMEOUT,
                decorrelated: bool = True,
                models=None, parent_ledger=None) -> _agent.Tool:
    """Build the `venice_review` Tool: a cold-context review of the session's own diff.

    `paid=False` mirrors `venice_scout`: a bounded nested model call, not a media
    purchase. The confirm gate guards side effects, and this has none.

    The per-session invocation budget is enforced HERE, in the closure, not in the
    prompt: a fix-review-fix spiral is the obvious failure mode of giving an agent a
    reviewer, and prose does not stop it. The counter is lock-guarded because
    `--parallel` can dispatch concurrently.

    What this tool returns is findings. It has no way to record that a diff was
    reviewed, and that is the point -- see the module docstring.
    """
    root = os.path.realpath(root)
    lock = threading.Lock()
    used = [0]

    def invoke(arguments, *, confirm: bool = False):
        args = _code._clean(arguments)
        with lock:
            if max_invocations > 0 and used[0] >= max_invocations:
                return _err(
                    f"review budget exhausted ({max_invocations} reviews this "
                    "session). Address the findings you already have; a further "
                    "review will not be run."
                )
            used[0] += 1
            remaining = max(0, max_invocations - used[0]) if max_invocations > 0 else None

        req_rounds = args.get("rounds")
        rounds = default_rounds if not isinstance(req_rounds, int) or req_rounds <= 0 \
            else req_rounds
        req_calls = args.get("max_tool_calls")
        calls = default_max_tool_calls if not isinstance(req_calls, int) or req_calls <= 0 \
            else req_calls
        paths = args.get("paths") if isinstance(args.get("paths"), list) else None
        try:
            base, _note = resolve_base(root, args.get("base") or None, exec_timeout)
            collected = collect_diff(root, base, paths=paths, exec_timeout=exec_timeout)
        except ReviewError as e:
            return _err(f"review: {e.message}")
        except Exception as e:                       # never take the parent loop down
            return _err(f"review failed: {e}")

        skip = should_skip(collected, "auto")
        if skip:
            out = skipped_envelope(collected, reason=skip, fail_on=DEFAULT_FAIL_ON)
            out["reviews_remaining"] = remaining
            return out
        try:
            result = run_cycle(
                oai, model, collected, base_kwargs, root=root, client=client,
                rounds=rounds, max_tool_calls=calls, max_tokens=max_tokens,
                focus=args.get("focus") or None, include_search=include_search,
                models=models, parent_ledger=parent_ledger,  # #117
            )
        except Exception as e:  # incl. openai.OpenAIError from the nested loop
            return _err(f"review failed: {e}")
        out = envelope(collected, result, model=model, decorrelated=decorrelated,
                       fail_on=DEFAULT_FAIL_ON)
        out["reviews_remaining"] = remaining
        return out

    return _agent.Tool(
        _agent.REVIEW_TOOL_NAME,
        "Review the current diff with a COLD-CONTEXT reviewer: a disposable subagent "
        "with a fresh context and only read-only tools, which did not write this code. "
        "Scoped to the diff against the default branch (including your uncommitted "
        "edits), never the whole repo. Returns defects with file:line, severity and a "
        "repro -- findings only; it does not approve, certify, or record anything. Use "
        "it before you hand work back, then fix what it found.",
        _REVIEW_SCHEMA, invoke, paid=False,
        category="agent", tags=("read", "review"),
    )
