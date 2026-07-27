"""`venice review` -- cold-context, diff-scoped code review (#80 part 1a).

Spins up a disposable reviewer on a FRESH context with only read-only tools, hands it
the diff against the repository's default branch (including uncommitted work), and
prints what it found: defects with `file:line`, a severity, a mechanism and a repro.

**This command produces findings. It does not certify anything.**

That distinction is the whole design, not a caveat. The exit code says what one run
found -- it is not a receipt, not an approval, and not a gate. An authoring agent may
ignore it or pass `--fail-on none`, and that is fine, because the exit code exists for
a human's shell and for the CI job of #80 part 1b, where the agent is not the invoker.
A gate the author can reach is not a gate; see `_review`'s module docstring for why
fusing the two operations would make part 1b unwinnable.

Nothing is written to disk: no receipt, no session, no state. `venice review` is
side-effect-free by construction, which is also why it has no `--auto`/`--manual`
mode gate and is safe to run unattended in a script.

Exit codes: 0 clean / skipped / empty diff, 1 findings at or above `--fail-on`,
2 bad input (not a repo, unknown --base, no key, missing `[openai]` extra), 6 unknown
`--model`, 10 no parseable REVIEW verdict, 130 Ctrl-C.
"""
from __future__ import annotations

import json
import os
import sys

from .. import auth, config, userconfig
from ..client import build_client_from_auth
from . import _agent, _models, _openai, _review


def register(subparsers) -> None:
    p = subparsers.add_parser(
        "review",
        help="Cold-context review of the current diff (findings only, no gate).",
        description=(
            "Review the diff between this branch and the repository's default branch "
            "-- including uncommitted changes -- with a disposable reviewer that has a "
            "fresh context, a different model where one is available, and only "
            "read-only tools. Reports defects with file:line and a repro. It produces "
            "FINDINGS ONLY: it approves nothing, certifies nothing, and writes nothing "
            "to disk."
        ),
    )
    p.add_argument(
        "focus", nargs="?",
        help="Optional hint about what to weigh most (e.g. 'the retry logic'). Not a "
        "hard scope -- the diff is always the scope.",
    )
    p.add_argument(
        "--root", default=None, metavar="DIR",
        help="Repository directory to review (default: current directory; also "
        "$VENICE_CODE_ROOT).",
    )
    p.add_argument(
        "--base", default=None, metavar="REF",
        help="Review against this git ref. Default: the repository's default branch, "
        "auto-detected (origin/HEAD, else origin/main, origin/master, main, master). "
        "The range is merge-base(REF, HEAD) compared against your working tree, so "
        "committed AND uncommitted work is included. Pass --base HEAD to review only "
        "uncommitted changes. Config: defaults.review.base.",
    )
    p.add_argument(
        "--path", action="append", dest="paths", default=None, metavar="PATH",
        help="Limit the diff to these paths (repeatable git pathspecs).",
    )
    p.add_argument(
        "--model", "-m", default=None,
        help="Reviewer model (must support function calling). Default: a "
        "function-calling model from a DIFFERENT family than the catalog default, so "
        "the reviewer's blind spots are not the author's. Config: defaults.review.model.",
    )
    p.add_argument("--temperature", "-t", type=float, default=None)
    p.add_argument("--max-tokens", type=int, default=None, dest="max_tokens")
    p.add_argument(
        "--json", action="store_true",
        help="Emit a JSON envelope (findings, verdict, base_sha/head_sha, model) to "
        "stdout instead of the human report.",
    )

    grp = p.add_argument_group("Review")
    grp.add_argument(
        "--rounds", type=int, default=None, metavar="N",
        help=f"Passes over the same diff, each told what the previous ones found "
        f"(default {_review.REVIEW_DEFAULT_ROUNDS}, max {_review.REVIEW_HARD_ROUNDS}). "
        f"Stops early when a pass finds nothing new. Config: defaults.review.rounds.",
    )
    grp.add_argument(
        "--effort", choices=_review.EFFORT_CHOICES, default=None,
        help="When to spend a model call: 'auto' (default) skips diffs with no code "
        "files -- docs, tests, lockfiles -- costing nothing; 'always' reviews "
        "regardless of surface; 'never' never calls the model. Note that 'auto' skips "
        "a test-only diff, so use 'always' to review changes to tests themselves. "
        "Config: defaults.review.effort.",
    )
    grp.add_argument(
        "--context", choices=_review.CONTEXT_CHOICES, default=None,
        dest="context",
        help="How much surrounding code to include: 'function' (default) expands each "
        "hunk to its enclosing function via git's -W; 'hunk' keeps plain context "
        "lines. Config: defaults.review.context.",
    )
    grp.add_argument(
        "--fail-on", choices=_review.FAIL_ON_CHOICES, default=None, dest="fail_on",
        help=f"Exit 1 when a finding at or above this severity is reported (default "
        f"{_review.DEFAULT_FAIL_ON}; 'minor' means any finding, 'none' always exits 0). "
        f"This reports what was found -- it is not a certification. Config: "
        f"defaults.review.fail_on.",
    )
    grp.add_argument(
        "--max-diff-chars", type=int, default=None, dest="max_diff_chars",
        metavar="N",
        help=f"Cap the diff handed to the reviewer (default "
        f"{_review.MAX_DIFF_CHARS}). Files past the cap are named in the report as "
        f"not covered. Config: defaults.review.max_diff_chars.",
    )
    grp.add_argument(
        "--max-tool-calls", type=int, default=None, dest="max_tool_calls",
        metavar="N",
        help=f"Cap the reviewer's own reads beyond the diff (default "
        f"{_review.REVIEW_MAX_TOOL_CALLS}, max {_review.REVIEW_HARD_CAP}). Config: "
        f"defaults.review.max_tool_calls.",
    )
    grp.add_argument(
        "--subagent-max-tokens", type=int, default=None, dest="subagent_max_tokens",
        metavar="N",
        help="Cap the cumulative prompt+completion tokens the reviewer spends across "
        "all rounds (default off). Config: defaults.review.subagent_max_tokens.",
    )
    grp.add_argument(
        "--exec-timeout", type=int, default=None, dest="exec_timeout",
        metavar="SECS", help="Timeout for the git commands that collect the diff.",
    )
    p.set_defaults(handler=_run)


def _gen_kwargs(args) -> dict:
    kw: dict = {}
    if args.temperature is not None:
        kw["temperature"] = args.temperature
    if args.max_tokens is not None:
        kw["max_tokens"] = args.max_tokens
    return kw


def _print_report(env: dict) -> None:
    """Human output: the report on stdout, provenance on stderr.

    Split deliberately so `venice review > findings.md` captures the findings and
    nothing else, while the operator still sees what was reviewed and with what.
    """
    print(f"base {env['base']} ({env['base_sha'][:12]}) -> HEAD "
          f"({env['head_sha'][:12]}); {len(env['files_reviewed'])} file(s); "
          f"model {env['model']}", file=sys.stderr)
    if env.get("report"):
        print(env["report"])
    outside = env.get("findings_outside_diff") or []
    if outside:
        print("\nReported outside the reviewed diff (not counted toward the exit "
              "code):", file=sys.stderr)
        for f in outside:
            print(f"  {f['file']}:{f['line']} [{f['severity']}] {f['summary']}",
                  file=sys.stderr)


def _run(args) -> int:
    userconfig.apply_defaults(args, "review")
    userconfig.apply_literals(
        args,
        rounds=_review.REVIEW_DEFAULT_ROUNDS,
        effort="auto",
        context="function",
        fail_on=_review.DEFAULT_FAIL_ON,
        max_diff_chars=_review.MAX_DIFF_CHARS,
        max_tool_calls=_review.REVIEW_MAX_TOOL_CALLS,
    )

    root = os.path.realpath(
        args.root or os.environ.get(config.ENV_CODE_ROOT) or os.getcwd())
    if not os.path.isdir(root):
        print(f"review: not a directory: {root}", file=sys.stderr)
        return 2

    exec_timeout = args.exec_timeout or _review._exec.DEFAULT_EXEC_TIMEOUT

    # Collect and triage BEFORE touching the network: a docs-only diff must cost zero
    # API calls and must not even require the [openai] extra (aiforge#19 SS5).
    try:
        base, note = _review.resolve_base(root, args.base, exec_timeout)
        if note:
            print(f"review: {note}", file=sys.stderr)
        collected = _review.collect_diff(
            root, base, paths=args.paths, context=args.context,
            max_chars=args.max_diff_chars, exec_timeout=exec_timeout,
        )
    except _review.ReviewError as e:
        print(f"review: {e.message}", file=sys.stderr)
        return e.code

    skip = _review.should_skip(collected, args.effort)
    if skip:
        env = _review.skipped_envelope(collected, reason=skip, fail_on=args.fail_on)
        if args.json:
            json.dump(env, sys.stdout, indent=2, default=str)
            sys.stdout.write("\n")
        else:
            print(f"review: {skip} -- no review run", file=sys.stderr)
        return 0

    openai = _openai.import_openai("review")
    if openai is None:
        return 2
    try:
        client = build_client_from_auth()
    except auth.AuthError as e:
        print(str(e), file=sys.stderr)
        return 2

    models = _models.catalog(client, "text")
    author = _models.default_model(models) or ""
    picked, decorrelated = _review.resolve_reviewer_model(models, args.model, author)
    model, rc = _models.resolve_model(
        picked, models, label="review", noun="text model",
        config_key="defaults.review.model",
    )
    if rc is not None:
        return rc
    ok, rc = _agent.check_function_calling(
        models, model, label="review",
        degraded_tail="venice review needs a tool-calling model (pass --model).",
        unverified_tail="attempting anyway",
        degrade=False,
    )
    if not ok:
        return rc
    if not decorrelated:
        print(f"review: reviewing with {model}, the same model family that would "
              "author here -- blind spots are correlated. Pass --model (or set "
              "defaults.review.model) to decorrelate.", file=sys.stderr)

    oai = _openai.build_openai(openai, client)
    try:
        result = _review.run_cycle(
            oai, model, collected, _gen_kwargs(args), root=root, client=client,
            rounds=args.rounds, max_tool_calls=args.max_tool_calls,
            max_tokens=args.subagent_max_tokens, focus=args.focus,
        )
    except openai.OpenAIError as e:
        return _openai.status_to_exit(openai, e, "review")
    except KeyboardInterrupt:
        print("\nreview: aborted", file=sys.stderr)
        return 130

    if result.get("status") != "ok":
        print(f"review: {result.get('message', 'review failed')}", file=sys.stderr)
        return 2

    env = _review.envelope(collected, result, model=model,
                           decorrelated=decorrelated, fail_on=args.fail_on)
    if args.json:
        json.dump(env, sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
    else:
        _print_report(env)

    if env["verdict"] == "unknown":
        print("review: could not parse a REVIEW verdict from the model (findings "
              "above may still be complete) -- exiting 10", file=sys.stderr)
        return 10
    return 1 if env["failed"] else 0
