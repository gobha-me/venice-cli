"""Unit tests for the cold-context reviewer -- #80 part 1a.

Covers `_review` (diff collection, surface triage, verdict/finding parsing, the
until-dry round loop, reviewer-model decorrelation, the `venice_review` rail tool)
and the `venice review` subcommand's exit codes.

Two things here are load-bearing beyond ordinary coverage:

1. **The separation pins** (`TestSeparationOfConcerns`). #80's whole design rests on
   "produce findings" and "certify a diff" staying separate operations, because the
   coding agent holds `apply_patch` + `shell` and would otherwise be able to write its
   own approval. These tests pin that structurally -- nothing on disk, no
   certification-shaped key anywhere in the output, no such parameter in the schema,
   no write/exec tool in the reviewer's grant, and no way to reach a reviewer through
   a worker. Each was mutation-tested when written.

2. **Real git fixtures.** `_exec.git_cmd` shells out, so a mocked git would only test
   the mock -- and `-z` numstat rename parsing plus `-W` function context are exactly
   what a mock would get wrong. Fixtures use plain `subprocess` (only the code *under
   test* goes through `git_cmd`) with the user's gitconfig neutralized.

No network, no real key, no sockets.
"""
import argparse
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from tests.test_chat import (
    FakeToolCompletion, _FnCall, _fake_openai_seq, FakeResp,
)

from venice.commands import _agent, _review, review

HAS_GIT = shutil.which("git") is not None

# Catalog with TWO function-calling models from different families, so the
# decorrelation happy path is reachable. `test_chat._text_payload`'s second model is
# deliberately non-FC, which only exercises the fallback.
_TWO_FAMILY_MODELS = [
    {"id": "llama-3.3-70b", "type": "text",
     "model_spec": {"traits": ["default"],
                    "capabilities": {"supportsFunctionCalling": True}}},
    {"id": "qwen3-235b", "type": "text",
     "model_spec": {"traits": [],
                    "capabilities": {"supportsFunctionCalling": True}}},
]


def _catalog_urlopen(models):
    payload = json.dumps({"object": "list", "data": models}).encode()

    def _u(req, timeout=None):
        return FakeResp(200, payload, "application/json")
    return _u


def _git(root, *argv):
    """Fixture-building git, deliberately NOT through `_exec.git_cmd`."""
    env = dict(os.environ)
    env.update({
        "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_SYSTEM": os.devnull,
        "GIT_AUTHOR_NAME": "T", "GIT_AUTHOR_EMAIL": "t@example.invalid",
        "GIT_COMMITTER_NAME": "T", "GIT_COMMITTER_EMAIL": "t@example.invalid",
    })
    subprocess.run(["git", *argv], cwd=root, env=env, check=True,
                   capture_output=True, text=True)


class _RepoBase(unittest.TestCase):
    """A throwaway repo: `master` with a baseline commit, then a `work` branch."""

    def setUp(self):
        if not HAS_GIT:
            self.skipTest("git is not installed")
        self.tmp = tempfile.mkdtemp()
        self.root = os.path.realpath(self.tmp)
        self.addCleanup(lambda: shutil.rmtree(self.tmp, ignore_errors=True))
        _git(self.root, "init", "-q", "-b", "master", ".")
        self._write("src/pool.cc",
                    "int acquire(int n) {\n  return n;\n}\n\n"
                    "void release(int* p) {\n  free(p);\n}\n")
        self._write("docs/guide.md", "hello\n")
        self._write("tests/test_pool.py", "def test_ok():\n    assert True\n")
        _git(self.root, "add", "-A")
        _git(self.root, "commit", "-qm", "baseline")
        _git(self.root, "checkout", "-qb", "work")

    def _write(self, rel, text):
        path = os.path.join(self.root, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)
        return path

    def _collect(self, **kw):
        base, _note = _review.resolve_base(self.root, kw.pop("base", None), 30)
        return _review.collect_diff(self.root, base, exec_timeout=30, **kw)


# --------------------------------------------------------------------------- #
# Surface triage
# --------------------------------------------------------------------------- #
class TestClassify(unittest.TestCase):
    CASES = [
        ("src/pool.cc", "code"), ("include/tf/widget.hpp", "code"),
        ("src/venice/commands/code.py", "code"), ("Makefile", "code"),
        ("src/app.rs", "code"), ("cmd/main.go", "code"),
        ("tests/test_pool.py", "test"), ("test/helper.cc", "test"),
        ("spec/thing_spec.rb", "test"), ("__tests__/a.js", "test"),
        ("src/pool_test.go", "test"), ("src/PoolTest.java", "test"),
        ("web/a.test.ts", "test"), ("web/b.spec.tsx", "test"),
        ("conftest.py", "test"), ("testdata/golden.txt", "test"),
        ("README.md", "docs"), ("docs/design.rst", "docs"),
        ("LICENSE", "docs"), ("CHANGELOG", "docs"), ("notes.txt", "docs"),
        ("CONTRIBUTING.md", "docs"),
        ("package-lock.json", "generated"), ("Cargo.lock", "generated"),
        ("go.sum", "generated"), ("web/app.min.js", "generated"),
        ("web/app.css.map", "generated"),
    ]

    def test_table(self):
        for path, expected in self.CASES:
            with self.subTest(path=path):
                self.assertEqual(_review.classify(path), expected)

    def test_binary_detected_from_numstat_markers(self):
        self.assertEqual(_review.classify("src/logo.png", "-", "-"), "binary")
        # ...and a normal count on the same path is NOT binary.
        self.assertNotEqual(_review.classify("src/logo.png", "3", "1"), "binary")

    def test_unknown_extensions_fall_through_to_code(self):
        # The safe direction: an unrecognised path costs a review rather than
        # silently skipping one.
        for path in ("src/thing.zig", "app/main.erl", "x/y/z"):
            with self.subTest(path=path):
                self.assertEqual(_review.classify(path), "code")


class TestNumstatParsing(unittest.TestCase):
    def test_plain_records(self):
        self.assertEqual(_review._parse_numstat_z("1\t2\ta.py\x005\t0\tb.py\x00"),
                         [("1", "2", "a.py"), ("5", "0", "b.py")])

    def test_rename_keeps_the_new_path(self):
        # `-z` renames are three NUL records: "adds\tdels\t", old, new.
        self.assertEqual(
            _review._parse_numstat_z("3\t1\t\x00old.py\x00new.py\x001\t1\tc.py\x00"),
            [("3", "1", "new.py"), ("1", "1", "c.py")])

    def test_binary_markers_survive(self):
        self.assertEqual(_review._parse_numstat_z("-\t-\timg.png\x00"),
                         [("-", "-", "img.png")])

    def test_empty_and_malformed_are_ignored(self):
        self.assertEqual(_review._parse_numstat_z(""), [])
        self.assertEqual(_review._parse_numstat_z("\x00\x00"), [])
        self.assertEqual(_review._parse_numstat_z("garbage\x00"), [])


# --------------------------------------------------------------------------- #
# Base resolution + diff collection
# --------------------------------------------------------------------------- #
class TestResolveBase(_RepoBase):
    def test_probes_to_master_when_origin_head_is_absent(self):
        # This is the COMMON shape, not an edge case: a repo created locally and
        # pushed has no refs/remotes/origin/HEAD at all, so both `symbolic-ref` and
        # `rev-parse --abbrev-ref origin/HEAD` fail and the candidate probe is what
        # actually resolves the base.
        base, note = _review.resolve_base(self.root, None, 30)
        self.assertEqual(base, "master")
        self.assertIn("auto-detected", note)

    def test_explicit_base_is_used_verbatim(self):
        base, note = _review.resolve_base(self.root, "HEAD", 30)
        self.assertEqual(base, "HEAD")
        self.assertEqual(note, "")

    def test_unknown_base_raises_exit_2(self):
        with self.assertRaises(_review.ReviewError) as cm:
            _review.resolve_base(self.root, "no-such-ref", 30)
        self.assertEqual(cm.exception.code, 2)
        self.assertIn("no-such-ref", cm.exception.message)

    def test_non_repo_raises_exit_2(self):
        outside = tempfile.mkdtemp()
        self.addCleanup(lambda: shutil.rmtree(outside, ignore_errors=True))
        with self.assertRaises(_review.ReviewError) as cm:
            _review.resolve_base(outside, None, 30)
        self.assertEqual(cm.exception.code, 2)
        self.assertIn("not a git repository", cm.exception.message)

    def test_repo_without_commits_raises(self):
        empty = os.path.realpath(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(empty, ignore_errors=True))
        _git(empty, "init", "-q", "-b", "master", ".")
        with self.assertRaises(_review.ReviewError) as cm:
            _review.resolve_base(empty, None, 30)
        self.assertIn("no commits", cm.exception.message)

    def test_falls_back_to_head_without_a_default_branch(self):
        _git(self.root, "branch", "-m", "master", "feature-only")
        base, note = _review.resolve_base(self.root, None, 30)
        self.assertEqual(base, "HEAD")
        self.assertIn("uncommitted", note)


class TestCollectDiff(_RepoBase):
    def test_default_range_spans_committed_and_uncommitted_work(self):
        # The load-bearing property of the default: `git diff <merge-base>` compares
        # against the WORKING TREE, so one range serves both the human reviewing a
        # branch before a PR and an agent reviewing edits it has not committed.
        self._write("src/pool.cc", "int acquire(int n) {\n  return n + 1;\n}\n")
        _git(self.root, "commit", "-qam", "committed change")
        self._write("src/extra.cc", "int extra() { return 0; }\n")   # uncommitted
        c = self._collect()
        self.assertIn("src/pool.cc", c["files_reviewed"])
        self.assertIn("src/extra.cc", c["files_reviewed"])

    def test_untracked_new_file_is_reviewed(self):
        # Regression pin for a bug this suite caught: `git diff` cannot see untracked
        # files, so a brand-new file the agent created but never staged was silently
        # invisible -- and a reviewer that skips whole new files while reporting
        # success is the worst failure mode this module has. Diffed against
        # /dev/null, which is how git renders a real new-file hunk anyway.
        self._write("src/brand_new.cc", "int fresh() {\n  return 42;\n}\n")
        c = self._collect()
        self.assertIn("src/brand_new.cc", c["files_reviewed"])
        self.assertIn("int fresh()", c["diff"])
        self.assertIn("new file", c["diff"])

    def test_untracked_file_is_still_triaged(self):
        # Untracked must not become a bypass around surface triage.
        self._write("docs/fresh.md", "notes\n")
        c = self._collect()
        self.assertEqual(c["files_reviewed"], [])
        self.assertEqual([s["reason"] for s in c["files_skipped"]], ["docs"])

    def test_gitignored_files_are_not_reviewed(self):
        # --exclude-standard: build output and venv noise must not reach the model.
        self._write(".gitignore", "build/\n")
        _git(self.root, "add", ".gitignore")
        _git(self.root, "commit", "-qm", "ignore build")
        self._write("build/generated.cc", "int junk() { return 0; }\n")
        c = self._collect()
        self.assertNotIn("build/generated.cc", c["changed_paths"])

    def test_base_head_narrows_to_uncommitted_only(self):
        self._write("src/pool.cc", "int acquire(int n) {\n  return n + 1;\n}\n")
        _git(self.root, "commit", "-qam", "committed change")
        self._write("src/extra.cc", "int extra() { return 0; }\n")
        c = self._collect(base="HEAD")
        self.assertEqual(c["files_reviewed"], ["src/extra.cc"])

    def test_provenance_shas_are_reported(self):
        # #80 part 1b binds a future check to a specific diff with exactly this pair.
        self._write("src/pool.cc", "int acquire(int n) {\n  return n + 2;\n}\n")
        c = self._collect()
        self.assertRegex(c["base_sha"], r"^[0-9a-f]{40}$")
        self.assertRegex(c["head_sha"], r"^[0-9a-f]{40}$")

    def test_merge_base_is_pinned_to_a_sha_not_a_moving_ref(self):
        # Re-deriving the range per round would let a moving origin/master silently
        # shift what is under review between passes.
        self._write("src/pool.cc", "int acquire(int n) {\n  return n + 3;\n}\n")
        c = self._collect()
        out = subprocess.run(["git", "merge-base", "master", "HEAD"], cwd=self.root,
                             capture_output=True, text=True)
        self.assertEqual(c["base_sha"], out.stdout.strip())

    def test_docs_and_tests_are_triaged_out_of_the_payload(self):
        self._write("docs/guide.md", "hello\nmore\n")
        self._write("tests/test_pool.py", "def test_ok():\n    assert False\n")
        self._write("src/pool.cc", "int acquire(int n) {\n  return n + 4;\n}\n")
        c = self._collect()
        self.assertEqual(c["files_reviewed"], ["src/pool.cc"])
        reasons = {s["path"]: s["reason"] for s in c["files_skipped"]}
        self.assertEqual(reasons["docs/guide.md"], "docs")
        self.assertEqual(reasons["tests/test_pool.py"], "test")
        # ...but every changed path is still reported, so a finding that names a
        # skipped file is not misfiled as "outside the diff".
        self.assertIn("docs/guide.md", c["changed_paths"])

    def test_function_context_expands_beyond_the_changed_line(self):
        # git's -W is what supplies "diff + enclosing functions" -- the reason no
        # hand-rolled (and necessarily Python-only) extractor exists in this repo.
        self._write("src/pool.cc",
                    "int acquire(int n) {\n  return n;\n}\n\n"
                    "void release(int* p) {\n  free(p);\n  p = 0;\n}\n")
        c = self._collect(context="function")
        self.assertIn("void release(int* p) {", c["diff"])

    def test_hunk_context_is_selectable(self):
        self._write("src/pool.cc", "int acquire(int n) {\n  return n + 5;\n}\n")
        c = self._collect(context="hunk")
        self.assertIn("acquire", c["diff"])

    def test_binary_file_is_skipped_not_embedded(self):
        with open(os.path.join(self.root, "logo.png"), "wb") as fh:
            fh.write(b"\x89PNG\r\n\x1a\n" + bytes(range(256)) * 4)
        c = self._collect()
        reasons = {s["path"]: s["reason"] for s in c["files_skipped"]}
        self.assertEqual(reasons.get("logo.png"), "binary")
        self.assertNotIn("logo.png", c["files_reviewed"])

    def test_rename_is_reviewed_under_its_new_path(self):
        _git(self.root, "mv", "src/pool.cc", "src/renamed.cc")
        c = self._collect()
        self.assertIn("src/renamed.cc", c["changed_paths"])

    def test_budget_does_not_systematically_starve_new_files(self):
        # Regression pin for a bias found by running this on a real feature branch:
        # untracked files are discovered after tracked ones, so a first-come budget
        # rule dropped both brand-new modules while keeping six edited files -- and
        # in an agent workflow a file that was just CREATED is the most important
        # thing in the diff. Selection is smallest-first, so a small new file wins
        # against a huge edited one regardless of listing order.
        self._write("src/pool.cc", "\n".join(
            f"int big{j}() {{ return {j}; }}" for j in range(400)) + "\n")
        self._write("src/brand_new.cc", "int tiny() { return 1; }\n")
        c = self._collect(max_chars=2000)
        self.assertIn("src/brand_new.cc", c["files_reviewed"])
        self.assertIn("src/pool.cc", c["files_omitted"])

    def test_file_order_is_independent_of_tracked_vs_untracked(self):
        self._write("src/aaa.cc", "int a() { return 1; }\n")     # untracked
        self._write("src/pool.cc", "int acquire(int n) {\n  return n + 30;\n}\n")
        c = self._collect()
        self.assertEqual(c["files_reviewed"], sorted(c["files_reviewed"]))

    def test_budget_omits_rather_than_silently_clipping(self):
        # A file dropped for budget must be NAMED, so the reviewer (and the operator)
        # know the review's coverage is partial.
        for i in range(4):
            self._write(f"src/big{i}.cc", "\n".join(f"int f{i}_{j}() {{ return {j}; }}"
                                                    for j in range(200)) + "\n")
        c = self._collect(max_chars=1500)
        self.assertTrue(c["files_omitted"])
        self.assertTrue(c["diff_truncated"])
        self.assertLessEqual(len(c["files_reviewed"]), 4)

    def test_empty_diff_yields_no_paths(self):
        c = self._collect()
        self.assertEqual(c["changed_paths"], [])
        self.assertEqual(c["diff"], "")

    def test_pathspec_narrows_the_diff(self):
        self._write("src/pool.cc", "int acquire(int n) {\n  return n + 6;\n}\n")
        self._write("src/other.cc", "int other() { return 1; }\n")
        c = self._collect(paths=["src/other.cc"])
        self.assertEqual(c["files_reviewed"], ["src/other.cc"])


class TestShouldSkip(_RepoBase):
    def test_docs_only_skips_with_a_reason(self):
        self._write("docs/guide.md", "hello\nmore\n")
        c = self._collect()
        reason = _review.should_skip(c, "auto")
        self.assertIsNotNone(reason)
        self.assertIn("no code files", reason)

    def test_effort_always_reviews_a_docs_only_diff(self):
        self._write("docs/guide.md", "hello\nmore\n")
        c = self._collect()
        self.assertIsNone(_review.should_skip(c, "always"))

    def test_effort_never_always_skips(self):
        self._write("src/pool.cc", "int acquire(int n) {\n  return n + 7;\n}\n")
        c = self._collect()
        self.assertIsNotNone(_review.should_skip(c, "never"))

    def test_empty_diff_skips_even_with_effort_always(self):
        c = self._collect()
        self.assertIn("empty diff", _review.should_skip(c, "always"))

    def test_code_change_is_not_skipped(self):
        self._write("src/pool.cc", "int acquire(int n) {\n  return n + 8;\n}\n")
        c = self._collect()
        self.assertIsNone(_review.should_skip(c, "auto"))


# --------------------------------------------------------------------------- #
# Verdict + findings parsing
# --------------------------------------------------------------------------- #
_REPORT = """SCOPE: read src/pool.cc and its caller.

FINDINGS:
src/pool.cc:142 [blocker] Freed buffer is reused after release()
WHY: release() frees the slab but the caller keeps its pointer.
REPRO: call acquire(), release(), then write through the old pointer at pool.cc:88.
FIX: null the caller-held pointer in release().
- src/pool.cc:207 [minor] off-by-one in the capacity check
WHY: uses <= where < is meant.
REPRO: fill to exactly capacity; the next insert overruns.
FIX: change <= to <.
src/pool.cc:301 severity omitted entirely
WHY: unstated.
src/elsewhere.cc:5 [major] a file the diff never touched
WHY: drive-by.

NOT CHECKED: the Windows path.

REVIEW: FINDINGS
"""


class TestVerdictParsing(unittest.TestCase):
    def test_clean_and_findings(self):
        self.assertEqual(_review.parse_verdict("REVIEW: CLEAN"), "clean")
        self.assertEqual(_review.parse_verdict("REVIEW: FINDINGS"), "findings")

    def test_markdown_and_case_tolerant(self):
        self.assertEqual(_review.parse_verdict("**review: clean**"), "clean")
        self.assertEqual(_review.parse_verdict("  Review:   FiNdInGs  "), "findings")

    def test_last_match_wins(self):
        # The model often restates the format before giving its verdict.
        self.assertEqual(
            _review.parse_verdict("say REVIEW: CLEAN or REVIEW: FINDINGS.\n"
                                  "REVIEW: CLEAN"), "clean")

    def test_absent_verdict_is_none(self):
        self.assertIsNone(_review.parse_verdict("no sentinel here"))
        self.assertIsNone(_review.parse_verdict(""))
        self.assertIsNone(_review.parse_verdict(None))


class TestFindingsParsing(unittest.TestCase):
    def setUp(self):
        self.changed = ["src/pool.cc", "src/app.cc"]
        self.inside, self.outside = _review.parse_findings(_REPORT, self.changed)

    def test_findings_are_located_and_ordered(self):
        self.assertEqual([(f["file"], f["line"]) for f in self.inside],
                         [("src/pool.cc", 142), ("src/pool.cc", 207),
                          ("src/pool.cc", 301)])

    def test_severity_is_extracted(self):
        self.assertEqual(self.inside[0]["severity"], "blocker")
        self.assertEqual(self.inside[1]["severity"], "minor")

    def test_missing_severity_fails_closed_to_major(self):
        # Failing OPEN would let sloppy formatting silently demote a blocker past
        # --fail-on. A noisy over-report is the safe direction.
        self.assertEqual(self.inside[2]["severity"], "major")

    def test_unknown_severity_also_fails_closed(self):
        self.assertEqual(_review._severity("catastrophic"), "major")
        self.assertEqual(_review._severity(None), "major")
        self.assertEqual(_review._severity(""), "major")
        self.assertEqual(_review._severity("MINOR"), "minor")

    def test_repro_lines_do_not_spawn_phantom_findings(self):
        # The REPRO for the first finding contains "pool.cc:88", which matches the
        # finding regex. Continuation lines must never start a new finding.
        self.assertNotIn(88, [f["line"] for f in self.inside])

    def test_block_keeps_why_repro_fix(self):
        block = self.inside[0]["block"]
        self.assertIn("WHY:", block)
        self.assertIn("REPRO:", block)
        self.assertIn("FIX:", block)

    def test_bullet_prefixes_are_tolerated(self):
        self.assertEqual(self.inside[1]["summary"],
                         "off-by-one in the capacity check")

    def test_finding_outside_the_diff_is_separated_not_dropped(self):
        # Not dropped: the reviewer may be right about a caller it read. But it must
        # not drive the exit code, or diff-scoping means nothing.
        self.assertEqual([f["file"] for f in self.outside], ["src/elsewhere.cc"])

    def test_line_range_is_accepted(self):
        ins, _ = _review.parse_findings(
            "FINDINGS:\nsrc/a.py:12-18 [major] spans lines\nREVIEW: FINDINGS",
            ["src/a.py"])
        self.assertEqual((ins[0]["file"], ins[0]["line"]), ("src/a.py", 12))

    def test_none_yields_no_findings(self):
        ins, out = _review.parse_findings(
            "SCOPE: all\nFINDINGS: none\nREVIEW: CLEAN", ["src/a.py"])
        self.assertEqual((ins, out), ([], []))

    def test_absolute_and_subdir_relative_paths_still_match(self):
        ins, out = _review.parse_findings(
            "FINDINGS:\n/abs/repo/src/pool.cc:9 [major] x\nREVIEW: FINDINGS",
            ["src/pool.cc"])
        self.assertEqual(len(ins), 1)
        self.assertEqual(out, [])

    def test_unparseable_prose_never_raises(self):
        ins, out = _review.parse_findings("FINDINGS:\ntotal gibberish ::::\n", ["a"])
        self.assertEqual((ins, out), ([], []))


class TestFailOn(unittest.TestCase):
    F = [{"severity": "minor"}, {"severity": "major"}]

    def test_thresholds(self):
        self.assertFalse(_review.fails(self.F, "none"))
        self.assertTrue(_review.fails(self.F, "minor"))
        self.assertTrue(_review.fails(self.F, "major"))
        self.assertFalse(_review.fails(self.F, "blocker"))

    def test_blocker_trips_every_level_but_none(self):
        f = [{"severity": "blocker"}]
        for level in ("minor", "major", "blocker"):
            self.assertTrue(_review.fails(f, level), level)
        self.assertFalse(_review.fails(f, "none"))

    def test_no_findings_never_fails(self):
        for level in _review.FAIL_ON_CHOICES:
            self.assertFalse(_review.fails([], level), level)


# --------------------------------------------------------------------------- #
# Reviewer-model decorrelation
# --------------------------------------------------------------------------- #
class TestDecorrelation(unittest.TestCase):
    def test_prefers_a_different_family(self):
        mid, dec = _review.resolve_reviewer_model(
            _TWO_FAMILY_MODELS, None, "llama-3.3-70b")
        self.assertEqual(mid, "qwen3-235b")
        self.assertTrue(dec)

    def test_same_family_different_id_is_not_decorrelation(self):
        # `qwen3-4b` and `qwen-2.5-coder` share blind spots; a different id alone
        # buys nothing, which is why the family heuristic exists.
        models = [
            {"id": "qwen3-4b", "type": "text",
             "model_spec": {"traits": ["default"],
                            "capabilities": {"supportsFunctionCalling": True}}},
            {"id": "qwen-2.5-coder", "type": "text",
             "model_spec": {"traits": [],
                            "capabilities": {"supportsFunctionCalling": True}}},
        ]
        mid, dec = _review.resolve_reviewer_model(models, None, "qwen3-4b")
        self.assertEqual(mid, "qwen-2.5-coder")
        self.assertFalse(dec)

    def test_single_model_catalog_falls_back_to_the_author(self):
        # Never a hard error: #80 says *prefer* a different model, and refusing would
        # make `venice code --review` unusable on a one-model deployment.
        models = [{"id": "solo-1", "type": "text",
                   "model_spec": {"traits": ["default"],
                                  "capabilities": {"supportsFunctionCalling": True}}}]
        mid, dec = _review.resolve_reviewer_model(models, None, "solo-1")
        self.assertEqual(mid, "solo-1")
        self.assertFalse(dec)

    def test_non_function_calling_models_are_not_offered(self):
        models = [
            {"id": "llama-3.3-70b", "type": "text",
             "model_spec": {"traits": ["default"],
                            "capabilities": {"supportsFunctionCalling": True}}},
            {"id": "qwen3-235b", "type": "text",
             "model_spec": {"traits": [],
                            "capabilities": {"supportsFunctionCalling": False}}},
        ]
        mid, dec = _review.resolve_reviewer_model(models, None, "llama-3.3-70b")
        self.assertEqual(mid, "llama-3.3-70b")
        self.assertFalse(dec)

    def test_explicit_request_wins(self):
        mid, dec = _review.resolve_reviewer_model(
            _TWO_FAMILY_MODELS, "qwen3-235b", "llama-3.3-70b")
        self.assertEqual(mid, "qwen3-235b")
        self.assertTrue(dec)

    def test_explicit_same_family_request_is_honoured_but_not_decorrelated(self):
        mid, dec = _review.resolve_reviewer_model(
            _TWO_FAMILY_MODELS, "llama-3.3-70b", "llama-3.3-70b")
        self.assertEqual(mid, "llama-3.3-70b")
        self.assertFalse(dec)

    def test_family_extraction(self):
        self.assertEqual(_review._family("qwen3-4b"), "qwen")
        self.assertEqual(_review._family("qwen-2.5-coder"), "qwen")
        self.assertEqual(_review._family("llama-3.3-70b"), "llama")
        self.assertEqual(_review._family("venice-uncensored"), "venice")


# --------------------------------------------------------------------------- #
# The round loop
# --------------------------------------------------------------------------- #
def _report(findings_lines, verdict="FINDINGS"):
    body = "\n".join(findings_lines) if findings_lines else "none"
    return f"SCOPE: x\n\nFINDINGS:\n{body}\n\nNOT CHECKED: y\n\nREVIEW: {verdict}\n"


class TestRunCycle(_RepoBase):
    def setUp(self):
        super().setUp()
        self._write("src/pool.cc", "int acquire(int n) {\n  return n + 9;\n}\n")
        self.collected = self._collect()

    def _cycle(self, reports, **kw):
        """Drive run_cycle with a stubbed run_review returning canned reports."""
        seen = []

        def _stub(oai, model, task, tools, base_kwargs, **kwargs):
            seen.append(task)
            return {"status": "ok", "report": reports[len(seen) - 1],
                    "tool_calls": 1, "truncated": False}

        with mock.patch.object(_agent, "run_review", _stub):
            out = _review.run_cycle(mock.MagicMock(), "m", self.collected, {},
                                    root=self.root, **kw)
        return out, seen

    def test_single_round_returns_findings(self):
        out, seen = self._cycle([_report(["src/pool.cc:1 [major] bad"])], rounds=1)
        self.assertEqual(out["verdict"], "findings")
        self.assertEqual(len(out["findings"]), 1)
        self.assertEqual(out["rounds"], 1)
        self.assertEqual(len(seen), 1)

    def test_second_round_is_told_what_the_first_found(self):
        out, seen = self._cycle(
            [_report(["src/pool.cc:1 [major] first"]),
             _report(["src/pool.cc:2 [major] second"])], rounds=2)
        self.assertEqual(len(seen), 2)
        # The entire first-round task, including the closing diff fence, is the
        # byte-identical prefix of round two. Prior findings may only follow it.
        self.assertTrue(seen[1].startswith(seen[0]))
        self.assertIn("ALREADY REPORTED", seen[1])
        self.assertIn("first", seen[1])
        self.assertNotIn("ALREADY REPORTED", seen[0])
        self.assertGreater(
            seen[1].index("ALREADY REPORTED"), seen[1].index("\n```\n")
        )
        self.assertEqual(len(out["findings"]), 2)

    def test_stops_early_when_a_round_adds_nothing_new(self):
        # Until-dry: a second pass that repeats itself must not buy a third.
        same = _report(["src/pool.cc:1 [major] same"])
        out, seen = self._cycle([same, same, same], rounds=3)
        self.assertEqual(len(seen), 2)
        self.assertEqual(len(out["findings"]), 1)
        self.assertEqual(out["rounds"], 2)

    def test_clean_first_round_does_not_buy_a_second(self):
        out, seen = self._cycle([_report([], "CLEAN")], rounds=3)
        self.assertEqual(len(seen), 1)
        self.assertEqual(out["verdict"], "clean")
        self.assertEqual(out["findings"], [])

    def test_rounds_are_clamped_to_the_hard_cap(self):
        reports = [_report([f"src/pool.cc:{i} [major] f{i}"]) for i in range(1, 10)]
        out, seen = self._cycle(reports, rounds=9)
        self.assertEqual(len(seen), _review.REVIEW_HARD_ROUNDS)
        self.assertEqual(out["rounds"], _review.REVIEW_HARD_ROUNDS)

    def test_system_prompt_is_byte_identical_across_rounds(self):
        # Prior findings ride in the TASK, never the system prompt: a stable prefix
        # is what keeps the input cache hot, and it keeps the prompt pins meaningful.
        systems = []

        def _stub(oai, model, task, tools, base_kwargs, **kwargs):
            systems.append(kwargs.get("system", _agent.REVIEW_SYSTEM))
            return {"status": "ok", "tool_calls": 0, "truncated": False,
                    "report": _report([f"src/pool.cc:{len(systems)} [major] f"])}

        with mock.patch.object(_agent, "run_review", _stub):
            _review.run_cycle(mock.MagicMock(), "m", self.collected, {},
                              root=self.root, rounds=2)
        self.assertEqual(len(set(systems)), 1)

    def test_unparseable_verdict_reprompts_once_then_gives_up(self):
        oai = mock.MagicMock()
        oai.chat.completions.create.return_value = FakeToolCompletion(
            content="I could not decide.")

        def _stub(*a, **kw):
            return {"status": "ok", "tool_calls": 0, "truncated": False,
                    "report": "SCOPE: x\nFINDINGS: none\n(no sentinel)"}

        with mock.patch.object(_agent, "run_review", _stub):
            out = _review.run_cycle(oai, "m", self.collected, {}, root=self.root,
                                    rounds=2)
        self.assertEqual(out["verdict"], "unknown")
        self.assertEqual(oai.chat.completions.create.call_count, 1)  # exactly one retry

    def test_reprompt_recovers_a_verdict(self):
        oai = mock.MagicMock()
        oai.chat.completions.create.return_value = FakeToolCompletion(
            content="REVIEW: CLEAN")

        def _stub(*a, **kw):
            return {"status": "ok", "tool_calls": 0, "truncated": False,
                    "report": "SCOPE: x\nFINDINGS: none\n(forgot the line)"}

        with mock.patch.object(_agent, "run_review", _stub):
            out = _review.run_cycle(oai, "m", self.collected, {}, root=self.root,
                                    rounds=1)
        self.assertEqual(out["verdict"], "clean")

    def test_verdict_retry_does_not_reuse_parent_cache_affinity(self):
        oai = mock.MagicMock()
        oai.chat.completions.create.return_value = FakeToolCompletion(
            content="REVIEW: CLEAN")

        def _stub(*a, **kw):
            return {"status": "ok", "tool_calls": 0, "truncated": False,
                    "report": "SCOPE: x\nFINDINGS: none\n(no sentinel)"}

        base = {"extra_body": {
            "prompt_cache_key": "parent-key",
            "venice_parameters": {"include_venice_system_prompt": False},
        }}
        with mock.patch.object(_agent, "run_review", _stub):
            out = _review.run_cycle(
                oai, "m", self.collected, base, root=self.root, rounds=1)
        self.assertEqual(out["verdict"], "clean")
        sent = oai.chat.completions.create.call_args.kwargs
        self.assertNotIn("prompt_cache_key", sent["extra_body"])
        self.assertEqual(
            sent["extra_body"]["venice_parameters"],
            {"include_venice_system_prompt": False},
        )
        self.assertEqual(
            base["extra_body"]["prompt_cache_key"], "parent-key")

    def test_verdict_retry_tokens_are_counted(self):
        # The retry runs outside `_run_disposable`, so nothing else records it. An
        # uncounted call is also an UNCAPPED one -- `--max-tokens` was silently not
        # covering a call it was meant to bound.
        oai = mock.MagicMock()
        oai.chat.completions.create.return_value = FakeToolCompletion(
            content="REVIEW: CLEAN",
            usage={"prompt_tokens": 700, "completion_tokens": 7},
        )

        def _stub(*a, **kw):
            kw["ledger"].record({"prompt_tokens": 10, "completion_tokens": 1})
            return {"status": "ok", "tool_calls": 0, "truncated": False,
                    "report": "SCOPE: x\nFINDINGS: none\n(forgot the line)"}

        with mock.patch.object(_agent, "run_review", _stub):
            out = _review.run_cycle(oai, "m", self.collected, {}, root=self.root,
                                    rounds=1)
        self.assertEqual(out["verdict"], "clean")   # the retry did its job...
        self.assertEqual(out["tokens"], 718)        # ...and paid for it: 11 + 707

    # --- #117: the cycle bills the parent's `review` bucket -----------------

    def _priced(self, usd):
        return [{"id": "reviewer", "model_spec": {"pricing": {
            "input": {"usd": usd}, "output": {"usd": usd * 2}}}}]

    def test_the_cycle_lands_in_the_parent_review_bucket(self):
        P = _agent.CostLedger()
        with mock.patch.object(_agent, "run_review", self._spender(1000, 500)):
            _review.run_cycle(mock.MagicMock(), "reviewer", self.collected, {},
                              root=self.root, rounds=1, models=self._priced(1.0),
                              parent_ledger=P)
        row = P.buckets["review"]
        self.assertEqual((row["calls"], row["prompt_tokens"]), (1, 1000))
        self.assertAlmostEqual(row["cost"], 0.0020)
        self.assertEqual(P.total, 0.0)              # never the parent's main loop

    def test_a_cycle_without_a_parent_ledger_is_unchanged(self):
        # `venice review` (the standalone CLI) passes neither argument.
        P_free, seen = self._cycle([_report(["src/pool.cc:1 [major] bad"])], rounds=1)
        self.assertEqual(P_free["verdict"], "findings")

    def test_the_reviewer_is_priced_at_its_own_model_rate(self):
        # THE reason pricing binds on the child. `--review-model` is deliberately a
        # DIFFERENT model from the author's -- decorrelation is the point of the rail --
        # so costing the reviewer's usage at the parent's rate would bill a fabricated
        # number. Parent at 1x, reviewer at 10x: the bucket must show the reviewer's.
        P = _agent.CostLedger()
        P.bind_pricing({"input": {"usd": 1.0}, "output": {"usd": 2.0}})
        with mock.patch.object(_agent, "run_review", self._spender(1000, 500)):
            _review.run_cycle(mock.MagicMock(), "reviewer", self.collected, {},
                              root=self.root, rounds=1, models=self._priced(10.0),
                              parent_ledger=P)
        self.assertAlmostEqual(P.buckets["review"]["cost"], 0.0200)   # 10x, not 0.0020

    def test_the_verdict_retry_is_billed_to_the_parent_too(self):
        # The retry is an API call made OUTSIDE `run_loop`. A usage callback threaded
        # into `run_loop` -- the shape #117 itself proposed -- would silently miss it.
        # Mirroring at `record()` catches it because that is where usage is READ.
        oai = mock.MagicMock()
        oai.chat.completions.create.return_value = FakeToolCompletion(
            content="REVIEW: CLEAN",
            usage={"prompt_tokens": 700, "completion_tokens": 7},
        )

        def _stub(*a, **kw):
            kw["ledger"].record({"prompt_tokens": 10, "completion_tokens": 1})
            return {"status": "ok", "tool_calls": 0, "truncated": False,
                    "report": "SCOPE: x\nFINDINGS: none\n(forgot the line)"}

        P = _agent.CostLedger()
        with mock.patch.object(_agent, "run_review", _stub):
            out = _review.run_cycle(oai, "reviewer", self.collected, {},
                                    root=self.root, rounds=1,
                                    models=self._priced(1.0), parent_ledger=P)
        self.assertEqual(out["verdict"], "clean")
        row = P.buckets["review"]
        self.assertEqual(row["calls"], 2)                  # the round AND the retry
        self.assertEqual(row["prompt_tokens"], 710)        # 10 + 700

    def test_every_round_of_one_cycle_bills_the_same_bucket(self):
        # ONE ledger spans the cycle, so N rounds are N calls in ONE bucket row.
        reports = [_report(["src/pool.cc:1 [major] a"]),
                   _report(["src/pool.cc:2 [major] b"])]
        seen = []

        def _stub(oai, model, task, tools, base_kwargs, **kw):
            kw["ledger"].record({"prompt_tokens": 100, "completion_tokens": 10})
            seen.append(task)
            return {"status": "ok", "report": reports[len(seen) - 1],
                    "tool_calls": 1, "truncated": False}

        P = _agent.CostLedger()
        with mock.patch.object(_agent, "run_review", _stub):
            _review.run_cycle(mock.MagicMock(), "reviewer", self.collected, {},
                              root=self.root, rounds=2, models=self._priced(1.0),
                              parent_ledger=P)
        self.assertEqual(sorted(P.buckets), ["review"])
        self.assertEqual(P.buckets["review"]["calls"], 2)

    @staticmethod
    def _spender(pt, ct):
        def _stub(oai, model, task, tools, base_kwargs, **kw):
            kw["ledger"].record({"prompt_tokens": pt, "completion_tokens": ct})
            return {"status": "ok", "tool_calls": 0, "truncated": False,
                    "report": _report([], "CLEAN")}
        return _stub

    def test_later_unreadable_round_does_not_erase_an_earlier_verdict(self):
        # Regression pin for a bug the drive suite caught: with the default 2 rounds,
        # a second pass that forgot the sentinel overwrote round 1's verdict with
        # "unknown", turning a review that had already found a blocker into exit 10
        # and discarding the findings entirely.
        oai = mock.MagicMock()
        oai.chat.completions.create.return_value = FakeToolCompletion(content="dunno")
        reports = [_report(["src/pool.cc:1 [blocker] real bug"]),
                   "SCOPE: x\nFINDINGS: none\n(no sentinel)"]
        seen = []

        def _stub(*a, **kw):
            seen.append(1)
            return {"status": "ok", "tool_calls": 0, "truncated": False,
                    "report": reports[len(seen) - 1]}

        with mock.patch.object(_agent, "run_review", _stub):
            out = _review.run_cycle(oai, "m", self.collected, {}, root=self.root,
                                    rounds=2)
        self.assertEqual(out["verdict"], "findings")
        self.assertEqual(len(out["findings"]), 1)

    def test_findings_override_a_contradictory_clean_sentinel(self):
        # A model that lists defects then types CLEAN is contradicting itself.
        # Trusting the content rather than the sentinel fails closed.
        out, _ = self._cycle(
            [_report(["src/pool.cc:1 [blocker] boom"], "CLEAN")], rounds=1)
        self.assertEqual(out["verdict"], "findings")

    def test_subagent_error_propagates_as_a_status(self):
        def _stub(*a, **kw):
            return {"status": "error", "message": "nested boom"}

        with mock.patch.object(_agent, "run_review", _stub):
            out = _review.run_cycle(mock.MagicMock(), "m", self.collected, {},
                                    root=self.root, rounds=2)
        self.assertEqual(out["status"], "error")
        self.assertIn("boom", out["message"])

    def test_findings_outside_the_diff_do_not_drive_the_exit_code(self):
        out, _ = self._cycle(
            [_report(["src/nowhere.cc:1 [blocker] not in this diff"])], rounds=1)
        self.assertEqual(out["findings"], [])
        self.assertEqual(len(out["findings_outside_diff"]), 1)
        self.assertFalse(_review.fails(out["findings"], "major"))


# --------------------------------------------------------------------------- #
# THE SEPARATION PINS -- #80's core constraint, structurally
# --------------------------------------------------------------------------- #
_CERT_RE = re.compile(r"approv|certif|receipt|signat|attest|sign|endorse|blessed",
                      re.I)


def _walk_keys(obj):
    """Every dict key at every depth."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k
            for sub in _walk_keys(v):
                yield sub
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            for sub in _walk_keys(item):
                yield sub


def _snapshot(*roots):
    """{path: (size, mtime)} for everything under `roots`."""
    seen = {}
    for root in roots:
        for dirpath, _dirs, files in os.walk(root):
            for name in files:
                p = os.path.join(dirpath, name)
                try:
                    st = os.stat(p)
                except OSError:
                    continue
                seen[p] = (st.st_size, st.st_mtime_ns)
    return seen


class TestSeparationOfConcerns(_RepoBase):
    """`venice review` produces findings and CANNOT certify.

    The coding agent holds `apply_patch` and `shell`. If certification were anything
    it could write -- a receipt file, an `approved` flag it could set -- it would
    eventually write it, not adversarially but because shortest-path-to-green is
    ordinary agent behaviour. Every test here pins one half of that boundary.
    """

    def setUp(self):
        super().setUp()
        self._write("src/pool.cc", "int acquire(int n) {\n  return n + 11;\n}\n")
        self.collected = self._collect()
        self.home = os.path.realpath(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(self.home, ignore_errors=True))

    def _full_run(self, stdout):
        """A complete `venice review --json` that produces findings."""
        args = _review_args(root=self.root, json=True)

        def _stub(*a, **kw):
            return {"status": "ok", "tool_calls": 1, "truncated": False,
                    "report": _report(["src/pool.cc:2 [blocker] boom"])}

        sess = os.path.join(self.home, "sessions")
        os.makedirs(sess, exist_ok=True)
        with mock.patch.dict(os.environ, {
                "VENICE_API_KEY": "fake", "HOME": self.home,
                "VENICE_SESSIONS_DIR": sess,
                "VENICE_MEMORY_DIR": os.path.join(self.home, "memory")}), \
             mock.patch("venice.userconfig.load_config",
                        lambda *a, **k: {"version": 1, "mcpServers": {},
                                         "defaults": {}}), \
             mock.patch("venice.client.urllib.request.urlopen",
                        _catalog_urlopen(_TWO_FAMILY_MODELS)), \
             mock.patch("openai.OpenAI", return_value=mock.MagicMock()), \
             mock.patch.object(_agent, "run_review", _stub), \
             mock.patch.object(sys, "stdout", stdout), \
             mock.patch.object(sys, "stderr", io.StringIO()):
            rc = review._run(args)
        return rc

    def test_review_writes_nothing_to_disk(self):
        """PIN 1: no receipt, no session, no state -- not one byte."""
        before = _snapshot(self.root, self.home)
        rc = self._full_run(io.StringIO())
        after = _snapshot(self.root, self.home)
        self.assertEqual(rc, 1)                      # it DID find a blocker...
        self.assertEqual(before, after)              # ...and still wrote nothing

    def test_review_envelope_has_no_certification_field(self):
        """PIN 2: nothing in the output can be read as an approval.

        This is the test that makes part 1b's tempting first draft -- "just add an
        `approved` boolean" -- fail loudly instead of quietly fusing the two
        operations that #80 exists to keep apart.
        """
        buf = io.StringIO()
        self._full_run(buf)
        env = json.loads(buf.getvalue())
        for key in _walk_keys(env):
            self.assertIsNone(_CERT_RE.search(key),
                              msg=f"certification-shaped key in envelope: {key}")
        self.assertEqual(set(env), {
            "status", "verdict", "findings", "findings_outside_diff", "report",
            "rounds", "base", "base_sha", "head_sha", "files_reviewed",
            "files_skipped", "files_omitted", "diff_truncated", "model",
            "decorrelated", "fail_on", "failed", "tokens", "token_cap",
            "tool_calls",
        })

    def test_tool_result_has_no_certification_field(self):
        """PIN 2b: the same, on the surface the coding agent actually sees."""
        def _stub(*a, **kw):
            return {"status": "ok", "tool_calls": 1, "truncated": False,
                    "report": _report(["src/pool.cc:2 [major] x"])}

        tool = _review.review_tool(mock.MagicMock(), "m", self.root, None, {})
        with mock.patch.object(_agent, "run_review", _stub):
            out = tool.invoke({})
        for key in _walk_keys(out):
            self.assertIsNone(_CERT_RE.search(key),
                              msg=f"certification-shaped key in tool result: {key}")

    def test_review_tool_schema_exposes_no_certify_parameter(self):
        """PIN 3: the agent cannot even ASK to have its work certified."""
        props = _review._REVIEW_SCHEMA.get("properties", {})
        for key in props:
            self.assertIsNone(_CERT_RE.search(key), msg=f"schema property: {key}")
        self.assertFalse(_review._REVIEW_SCHEMA.get("required"))
        # `model` is absent too: operator-controlled, so the agent cannot escalate
        # itself onto a costlier reviewer.
        self.assertNotIn("model", props)

    def test_review_grants_no_write_or_exec_tool(self):
        """PIN 4: the reviewer's own toolset cannot mutate anything."""
        granted = {}

        def _stub(oai, model, task, tools, base_kwargs, **kw):
            granted["tools"] = tools
            return {"status": "ok", "tool_calls": 0, "truncated": False,
                    "report": _report([], "CLEAN")}

        tool = _review.review_tool(mock.MagicMock(), "m", self.root, None, {})
        with mock.patch.object(_agent, "run_review", _stub):
            tool.invoke({})
        names = {t.name for t in granted["tools"]}
        self.assertTrue(all(t.paid is False for t in granted["tools"]))
        self.assertTrue(names.isdisjoint({
            "write_file", "edit_file", "apply_patch", "run", "attach_root",
            "reindex", _agent.SCOUT_TOOL_NAME, _agent.SPAWN_TOOL_NAME,
            _agent.MERGE_TOOL_NAME, _agent.REVIEW_TOOL_NAME,
        }))
        self.assertLessEqual({"read_file", "grep", "git"}, names)

    def test_no_nested_review(self):
        """PIN 5: nesting stays capped at one level across the new edge."""
        schema = {"type": "object", "properties": {}}
        rev = _agent.Tool(_agent.REVIEW_TOOL_NAME, "d", schema,
                          lambda a, *, confirm=False: {}, paid=False)
        for fn in (_agent.run_scout, _agent.run_spawn, _agent.run_review):
            with self.subTest(guard=fn.__name__):
                with self.assertRaises(ValueError):
                    fn(None, "m", "t", [rev], {}, max_tool_calls=3)

    def _spawn_grant(self, role="code"):
        """The tool names a worker actually receives, with a reviewer in the parent."""
        from venice.commands import _code
        tools = _code.code_tools(self.root, None, exec_timeout=30)
        tools.append(_review.review_tool(mock.MagicMock(), "m", self.root, None, {}))
        granted = {}

        def _stub(oai, model, task, tools_, base_kwargs, **kw):
            granted["names"] = {t.name for t in tools_}
            return {"status": "ok", "tool_calls": 0, "truncated": False, "report": "r"}

        spawn = _code.spawn_tool(mock.MagicMock(), "m", {}, tools)
        with mock.patch.object(_agent, "run_spawn", _stub):
            spawn.invoke({"task": "do a thing", "role": role})
        return granted.get("names", set())

    # PIN 6 is three INDEPENDENT barriers between a worker and the reviewer. The
    # end-to-end check below passes if any one of them holds, which makes it useless
    # for pinning a particular one -- removing the `category != "agent"` filter alone
    # left it green. So each barrier gets a test that fails when that barrier alone
    # is removed, and the end-to-end case is kept for what it actually is.
    def test_agent_cannot_reach_review_through_spawn(self):
        """PIN 6 (end-to-end): a worker's grant never contains the reviewer."""
        self.assertNotIn(_agent.REVIEW_TOOL_NAME, self._spawn_grant())

    def test_no_spawn_role_grants_the_agent_category(self):
        """PIN 6a: the role->category presets never request `agent`."""
        from venice.commands import _code
        for role, cats in _code._ROLE_CATEGORIES.items():
            with self.subTest(role=role):
                self.assertNotIn("agent", cats)

    def test_spawn_filter_excludes_agent_category_even_if_a_role_asked_for_it(self):
        """PIN 6b: the explicit `category != "agent"` filter, on its own.

        No shipped role requests `agent`, so this barrier is unreachable through the
        presets -- which is exactly why it needs a direct test. Patch a role to ask
        for `agent` and the filter must still refuse.
        """
        from venice.commands import _code
        greedy = dict(_code._ROLE_CATEGORIES)
        greedy["code"] = set(greedy["code"]) | {"agent"}
        with mock.patch.object(_code, "_ROLE_CATEGORIES", greedy):
            names = self._spawn_grant()
        self.assertNotIn(_agent.REVIEW_TOOL_NAME, names)
        self.assertNotIn(_agent.SCOUT_TOOL_NAME, names)
        self.assertIn("read_file", names)   # anti-vacuity: the grant is non-empty

    def test_review_tool_is_not_in_the_registry(self):
        """The rail invariant: review adds no advertised/registry tool."""
        self.assertNotIn(_agent.REVIEW_TOOL_NAME, _agent.select())
        self.assertNotIn(_agent.REVIEW_TOOL_NAME,
                         {t.name for t in _agent.builtin_tools(None)})


# --------------------------------------------------------------------------- #
# The `venice_review` rail tool
# --------------------------------------------------------------------------- #
class TestReviewTool(_RepoBase):
    def setUp(self):
        super().setUp()
        self._write("src/pool.cc", "int acquire(int n) {\n  return n + 12;\n}\n")

    def _tool(self, **kw):
        return _review.review_tool(mock.MagicMock(), "m", self.root, None, {}, **kw)

    def test_descriptor_shape(self):
        t = self._tool()
        self.assertEqual(t.name, _agent.REVIEW_TOOL_NAME)
        self.assertFalse(t.paid)
        self.assertEqual(t.category, "agent")   # keeps it out of every worker grant
        self.assertIn("review", t.tags)

    def test_not_a_default_code_tools_rail(self):
        from venice.commands import _code
        names = {t.name for t in _code.code_tools(self.root, None, exec_timeout=30)}
        self.assertNotIn(_agent.REVIEW_TOOL_NAME, names)

    def test_budget_is_structural_and_exhausts(self):
        # Prompt text would not stop a fix-review-fix spiral; the counter does.
        def _stub(*a, **kw):
            return {"status": "ok", "tool_calls": 0, "truncated": False,
                    "report": _report(["src/pool.cc:1 [major] x"])}

        t = self._tool(max_invocations=2)
        with mock.patch.object(_agent, "run_review", _stub):
            first = t.invoke({})
            second = t.invoke({})
            third = t.invoke({})
        self.assertEqual(first["status"], "ok")
        self.assertEqual(first["reviews_remaining"], 1)
        self.assertEqual(second["reviews_remaining"], 0)
        self.assertEqual(third["status"], "error")
        self.assertIn("budget exhausted", third["message"])

    def test_tool_call_budget_is_clamped(self):
        got = {}

        def _stub(oai, model, task, tools, base_kwargs, **kw):
            got["calls"] = kw.get("max_tool_calls")
            return {"status": "ok", "tool_calls": 0, "truncated": False,
                    "report": _report([], "CLEAN")}

        with mock.patch.object(_agent, "run_review", _stub):
            self._tool().invoke({"max_tool_calls": 9999})
        self.assertEqual(got["calls"], _review.REVIEW_HARD_CAP)

    def test_the_tool_forwards_the_parent_ledger_to_the_cycle(self):
        # #117: the rail is TWO layers -- `review_tool` builds nothing, `run_cycle` owns
        # the ledger. Testing `run_cycle` directly (as the cycle tests do) leaves the
        # forwarding hop unguarded: a mutation dropping `parent_ledger=` from the
        # `run_cycle` call here survived the whole suite until this test existed.
        def _stub(oai, model, task, tools, base_kwargs, **kw):
            kw["ledger"].record({"prompt_tokens": 1000, "completion_tokens": 500})
            return {"status": "ok", "tool_calls": 0, "truncated": False,
                    "report": _report([], "CLEAN")}

        P = _agent.CostLedger()
        models = [{"id": "m", "model_spec": {"pricing": {
            "input": {"usd": 1.0}, "output": {"usd": 2.0}}}}]
        with mock.patch.object(_agent, "run_review", _stub):
            out = self._tool(models=models, parent_ledger=P).invoke({})
        self.assertEqual(out["status"], "ok")
        self.assertIn("review", P.buckets)          # a clean FAIL, not a KeyError
        self.assertEqual(P.buckets["review"]["prompt_tokens"], 1000)
        self.assertAlmostEqual(P.buckets["review"]["cost"], 0.0020)

    def test_the_tool_still_works_with_no_parent_ledger(self):
        def _stub(oai, model, task, tools, base_kwargs, **kw):
            return {"status": "ok", "tool_calls": 0, "truncated": False,
                    "report": _report([], "CLEAN")}

        with mock.patch.object(_agent, "run_review", _stub):
            self.assertEqual(self._tool().invoke({})["status"], "ok")

    def test_docs_only_diff_short_circuits_without_a_model_call(self):
        _git(self.root, "checkout", "-q", "--", "src/pool.cc")
        self._write("docs/guide.md", "hello\nmore\n")
        called = []

        def _stub(*a, **kw):
            called.append(1)
            return {"status": "ok", "report": "", "tool_calls": 0, "truncated": False}

        with mock.patch.object(_agent, "run_review", _stub):
            out = self._tool().invoke({})
        self.assertEqual(out["status"], "skipped")
        self.assertEqual(called, [])

    def test_nested_error_becomes_an_error_envelope(self):
        def _boom(*a, **kw):
            raise RuntimeError("nested explosion")

        with mock.patch.object(_agent, "run_review", _boom):
            out = self._tool().invoke({})
        self.assertEqual(out["status"], "error")
        self.assertIn("nested explosion", out["message"])

    def test_non_repo_root_returns_an_error_not_a_traceback(self):
        outside = os.path.realpath(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(outside, ignore_errors=True))
        t = _review.review_tool(mock.MagicMock(), "m", outside, None, {})
        out = t.invoke({})
        self.assertEqual(out["status"], "error")
        self.assertIn("not a git repository", out["message"])


# --------------------------------------------------------------------------- #
# `venice review` exit codes
# --------------------------------------------------------------------------- #
def _review_args(**ov):
    base = dict(
        focus=None, root=None, base=None, paths=None, model=None, temperature=None,
        max_tokens=None, json=False, rounds=None, effort=None, context=None,
        fail_on=None, max_diff_chars=None, max_tool_calls=None,
        subagent_max_tokens=None, exec_timeout=None,
    )
    base.update(ov)
    return argparse.Namespace(**base)


class TestReviewCommand(_RepoBase):
    def _run(self, args, report=None, models=None, stdout=None, stderr=None,
             run_review=None):
        stub = run_review or (lambda *a, **kw: {
            "status": "ok", "tool_calls": 1, "truncated": False,
            "report": report if report is not None else _report([], "CLEAN")})
        with mock.patch.dict(os.environ, {"VENICE_API_KEY": "fake"}), \
             mock.patch("venice.userconfig.load_config",
                        lambda *a, **k: {"version": 1, "mcpServers": {},
                                         "defaults": {}}), \
             mock.patch("venice.client.urllib.request.urlopen",
                        _catalog_urlopen(models or _TWO_FAMILY_MODELS)), \
             mock.patch("openai.OpenAI", return_value=mock.MagicMock()), \
             mock.patch.object(_agent, "run_review", stub), \
             mock.patch.object(sys, "stdout", stdout or io.StringIO()), \
             mock.patch.object(sys, "stderr", stderr or io.StringIO()):
            return review._run(args)

    def test_clean_review_exits_zero(self):
        self._write("src/pool.cc", "int acquire(int n) {\n  return n + 13;\n}\n")
        self.assertEqual(self._run(_review_args(root=self.root)), 0)

    def test_findings_at_threshold_exit_one(self):
        self._write("src/pool.cc", "int acquire(int n) {\n  return n + 14;\n}\n")
        rc = self._run(_review_args(root=self.root),
                       report=_report(["src/pool.cc:1 [blocker] boom"]))
        self.assertEqual(rc, 1)

    def test_findings_below_threshold_exit_zero(self):
        self._write("src/pool.cc", "int acquire(int n) {\n  return n + 15;\n}\n")
        rc = self._run(_review_args(root=self.root, fail_on="blocker"),
                       report=_report(["src/pool.cc:1 [minor] nit"]))
        self.assertEqual(rc, 0)

    def test_fail_on_none_always_exits_zero(self):
        self._write("src/pool.cc", "int acquire(int n) {\n  return n + 16;\n}\n")
        rc = self._run(_review_args(root=self.root, fail_on="none"),
                       report=_report(["src/pool.cc:1 [blocker] boom"]))
        self.assertEqual(rc, 0)

    def test_unparseable_verdict_exits_ten(self):
        self._write("src/pool.cc", "int acquire(int n) {\n  return n + 17;\n}\n")
        oai = mock.MagicMock()
        oai.chat.completions.create.return_value = FakeToolCompletion(content="dunno")
        with mock.patch("openai.OpenAI", return_value=oai):
            rc = self._run(_review_args(root=self.root),
                           report="SCOPE: x\nFINDINGS: none\n(no sentinel)")
        self.assertEqual(rc, 10)

    def test_docs_only_exits_zero_without_importing_the_sdk(self):
        # The cost-discipline claim, enforced: triage runs before the SDK import, so
        # a docs-only diff needs neither the [openai] extra nor an API key.
        self._write("docs/guide.md", "hello\nmore\n")
        called = []
        with mock.patch("venice.commands._openai.import_openai",
                        lambda *a, **k: called.append(1)), \
             mock.patch.object(sys, "stdout", io.StringIO()), \
             mock.patch.object(sys, "stderr", io.StringIO()):
            rc = review._run(_review_args(root=self.root))
        self.assertEqual(rc, 0)
        self.assertEqual(called, [])

    def test_empty_diff_exits_zero(self):
        with mock.patch.object(sys, "stdout", io.StringIO()), \
             mock.patch.object(sys, "stderr", io.StringIO()):
            rc = review._run(_review_args(root=self.root))
        self.assertEqual(rc, 0)

    def test_unknown_base_exits_two(self):
        with mock.patch.object(sys, "stderr", io.StringIO()):
            rc = review._run(_review_args(root=self.root, base="nope"))
        self.assertEqual(rc, 2)

    def test_non_repo_exits_two(self):
        outside = os.path.realpath(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(outside, ignore_errors=True))
        with mock.patch.object(sys, "stderr", io.StringIO()):
            rc = review._run(_review_args(root=outside))
        self.assertEqual(rc, 2)

    def test_unknown_model_exits_six(self):
        self._write("src/pool.cc", "int acquire(int n) {\n  return n + 18;\n}\n")
        err = io.StringIO()
        rc = self._run(_review_args(root=self.root, model="no-such-model"),
                       stderr=err)
        self.assertEqual(rc, 6)
        self.assertIn("unknown text model", err.getvalue())

    def test_json_envelope_carries_provenance(self):
        self._write("src/pool.cc", "int acquire(int n) {\n  return n + 19;\n}\n")
        out = io.StringIO()
        self._run(_review_args(root=self.root, json=True),
                  report=_report(["src/pool.cc:1 [major] x"]), stdout=out)
        env = json.loads(out.getvalue())
        self.assertEqual(env["verdict"], "findings")
        self.assertRegex(env["base_sha"], r"^[0-9a-f]{40}$")
        self.assertRegex(env["head_sha"], r"^[0-9a-f]{40}$")
        self.assertEqual(env["model"], "qwen3-235b")   # decorrelated by default
        self.assertTrue(env["decorrelated"])
        self.assertTrue(env["failed"])

    def test_same_family_catalog_warns_but_still_reviews(self):
        solo = [{"id": "llama-3.3-70b", "type": "text",
                 "model_spec": {"traits": ["default"],
                                "capabilities": {"supportsFunctionCalling": True}}}]
        self._write("src/pool.cc", "int acquire(int n) {\n  return n + 20;\n}\n")
        err = io.StringIO()
        rc = self._run(_review_args(root=self.root), models=solo, stderr=err)
        self.assertEqual(rc, 0)                       # never a hard error
        self.assertIn("blind spots are correlated", err.getvalue())

    def test_human_output_puts_findings_on_stdout_and_notes_on_stderr(self):
        # `venice review > findings.md` must capture the findings and nothing else.
        self._write("src/pool.cc", "int acquire(int n) {\n  return n + 21;\n}\n")
        out, err = io.StringIO(), io.StringIO()
        self._run(_review_args(root=self.root),
                  report=_report(["src/pool.cc:1 [major] x"]), stdout=out, stderr=err)
        self.assertIn("src/pool.cc:1", out.getvalue())
        self.assertIn("base master", err.getvalue())
        self.assertNotIn("base master", out.getvalue())


# --------------------------------------------------------------------------- #
# `venice code --review` wiring
# --------------------------------------------------------------------------- #
class TestReviewRailWiring(_RepoBase):
    """`venice code --review` folds the rail in and keeps it operator-controlled."""

    def _run(self, args, seq, models=None, stderr=None, config=None):
        from venice.commands import code
        fake, calls = _fake_openai_seq(seq)
        stdin = mock.MagicMock()
        stdin.isatty.return_value = False
        sess = tempfile.mkdtemp()
        self.addCleanup(lambda: shutil.rmtree(sess, ignore_errors=True))
        with mock.patch.dict(os.environ, {"VENICE_API_KEY": "fake",
                                          "VENICE_SESSIONS_DIR": sess}), \
             mock.patch("venice.userconfig.load_config",
                        lambda *a, **k: {"version": 1, "mcpServers": {},
                                         "defaults": config or {}}), \
             mock.patch("venice.client.urllib.request.urlopen",
                        _catalog_urlopen(models or _TWO_FAMILY_MODELS)), \
             mock.patch("openai.OpenAI", return_value=fake), \
             mock.patch.object(sys, "stdin", stdin), \
             mock.patch.object(sys, "stdout", io.StringIO()), \
             mock.patch.object(sys, "stderr", stderr or io.StringIO()):
            rc = code._run(args)
        return rc, calls

    def _args(self, **ov):
        base = dict(
            task="do x", root=self.root, model=None, system=None, temperature=None,
            max_tokens=None, json=False, auto=None, manual=None, yes=None,
            plan_only=True, no_plan=False, no_verify=False, max_tool_calls=None,
            exec_timeout=None, interactive=False, resume=None, assets=None,
            auto_compact=None, compact_threshold=None, compact_keep_turns=None,
            session_max_spend=None, cache_guard=None, cont=None, ephemeral=None,
            review=None, review_model=None, review_rounds=None,
        )
        base.update(ov)
        return argparse.Namespace(**base)

    def test_review_flag_advertises_the_tool_in_the_prompt(self):
        seq = [FakeToolCompletion("1. do it\nAcceptance criteria:\n- ok")]
        rc, calls = self._run(self._args(review=True), seq)
        self.assertEqual(rc, 0)
        self.assertIn(_agent.REVIEW_TOOL_NAME, calls[0]["messages"][0]["content"])

    def test_without_the_flag_the_tool_is_absent(self):
        seq = [FakeToolCompletion("1. do it\nAcceptance criteria:\n- ok")]
        rc, calls = self._run(self._args(), seq)
        self.assertEqual(rc, 0)
        self.assertNotIn(_agent.REVIEW_TOOL_NAME, calls[0]["messages"][0]["content"])

    def test_reviewer_defaults_to_a_different_family_than_the_author(self):
        built = {}
        real = _review.review_tool

        def _spy(oai, model, *a, **kw):
            built["model"] = model
            built["decorrelated"] = kw.get("decorrelated")
            return real(oai, model, *a, **kw)

        seq = [FakeToolCompletion("1. do it\nAcceptance criteria:\n- ok")]
        with mock.patch.object(_review, "review_tool", _spy):
            self._run(self._args(review=True), seq)
        self.assertEqual(built["model"], "qwen3-235b")   # author is llama-3.3-70b
        self.assertTrue(built["decorrelated"])

    def test_explicit_review_model_wins(self):
        built = {}
        real = _review.review_tool

        def _spy(oai, model, *a, **kw):
            built["model"] = model
            return real(oai, model, *a, **kw)

        seq = [FakeToolCompletion("1. do it\nAcceptance criteria:\n- ok")]
        with mock.patch.object(_review, "review_tool", _spy):
            self._run(self._args(review=True, review_model="llama-3.3-70b"), seq)
        self.assertEqual(built["model"], "llama-3.3-70b")

    def test_unknown_config_reviewer_fails_before_a_completion_with_recovery(self):
        err = io.StringIO()
        rc, calls = self._run(
            self._args(review=True), [], stderr=err,
            config={"code": {"review_model": "retired-reviewer"}},
        )
        self.assertEqual(rc, 6)
        self.assertEqual(calls, [])
        self.assertIn("unknown review model 'retired-reviewer'", err.getvalue())
        self.assertIn(
            "venice config unset defaults.code.review_model", err.getvalue(),
        )

    def test_single_family_catalog_warns_instead_of_failing(self):
        solo = [{"id": "llama-3.3-70b", "type": "text",
                 "model_spec": {"traits": ["default"],
                                "capabilities": {"supportsFunctionCalling": True}}}]
        seq = [FakeToolCompletion("1. do it\nAcceptance criteria:\n- ok")]
        err = io.StringIO()
        rc, _ = self._run(self._args(review=True), seq, models=solo, stderr=err)
        self.assertEqual(rc, 0)
        self.assertIn("blind spots are correlated", err.getvalue())

    def test_review_rounds_threads_into_the_factory(self):
        built = {}
        real = _review.review_tool

        def _spy(oai, model, *a, **kw):
            built.update(kw)
            return real(oai, model, *a, **kw)

        seq = [FakeToolCompletion("1. do it\nAcceptance criteria:\n- ok")]
        with mock.patch.object(_review, "review_tool", _spy):
            self._run(self._args(review=True, review_rounds=3), seq)
        self.assertEqual(built["default_rounds"], 3)

    def test_rail_does_not_disturb_the_registry_pins(self):
        # The whole reason review is a factory-built rail rather than a ToolSpec.
        self.assertEqual(len(_agent.select()), 16)
        self.assertEqual(len(_agent.builtin_tools(None)), 14)


if __name__ == "__main__":
    unittest.main()
