"""Unit tests for the shared shell/exec rails + allow/deny policy (`_exec`, #33).

Hermetic: policy checks are pure; the run/git tests exec a real /bin/sh against a
throwaway tmpdir (no network, no real key). stdlib-only, runs on the 3.9 floor.
"""
import os
import tempfile
import unittest
from unittest import mock

from venice.commands import _exec


class TestCheckPolicy(unittest.TestCase):
    def test_no_policy_allows_anything(self):
        self.assertIsNone(_exec.check_policy("anything | goes; here", allow=[], deny=[]))

    def test_allowlist_permits_listed_leading_command(self):
        self.assertIsNone(_exec.check_policy("git status -s", allow=["git"], deny=[]))

    def test_allowlist_matches_basename_of_absolute_path(self):
        self.assertIsNone(_exec.check_policy("/usr/bin/git log", allow=["git"], deny=[]))

    def test_allowlist_globs_on_leading_token(self):
        self.assertIsNone(_exec.check_policy("python3 x.py", allow=["python*"], deny=[]))

    def test_allowlist_rejects_unlisted_command(self):
        msg = _exec.check_policy("rm file", allow=["ls", "git"], deny=[])
        self.assertIsNotNone(msg)
        self.assertIn("not in the shell allowlist", msg)

    def test_allowlist_rejects_operators_even_if_argv0_allowed(self):
        # The core trap: leading token is allowlisted but a chained command isn't.
        msg = _exec.check_policy("gh pr view && rm -rf ~", allow=["gh"], deny=[])
        self.assertIsNotNone(msg)
        self.assertIn("single simple command", msg)

    def test_allowlist_rejects_pipe_redirect_subst(self):
        for cmd in ("ls | sh", "cat x > y", "echo $(whoami)", "echo `id`", "a; b"):
            with self.subTest(cmd=cmd):
                self.assertIsNotNone(
                    _exec.check_policy(cmd, allow=["ls", "cat", "echo", "a"], deny=[])
                )

    def test_deny_always_enforced_even_without_allowlist(self):
        msg = _exec.check_policy("echo hi; sudo reboot", allow=[], deny=["sudo"])
        self.assertIsNotNone(msg)
        self.assertIn("deny", msg)

    def test_deny_wins_over_allow(self):
        msg = _exec.check_policy("rm -rf /", allow=["rm"], deny=["rm"])
        self.assertIsNotNone(msg)
        self.assertIn("deny", msg)

    def test_deny_substring_glob_on_full_string(self):
        msg = _exec.check_policy("gh pr merge && rm -rf ~", allow=[], deny=["*rm -rf*"])
        self.assertIsNotNone(msg)

    def test_deny_matches_token_inside_chain(self):
        # `rm` appears only inside a chain; token-level matching still catches it.
        msg = _exec.check_policy("build && rm out", allow=[], deny=["rm"])
        self.assertIsNotNone(msg)


class TestRunCmd(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.root = os.path.realpath(self.tmp)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_blocked_command_errors_before_confirm(self):
        # A denied command must NOT return a confirmation gate -- it can never be
        # approved, so it's refused up front (before the [y/N] gate).
        r = _exec.run_cmd(self.root, "sudo reboot", deny=["sudo"])
        self.assertEqual(r["status"], "error")
        self.assertIn("deny", r["message"])

    def test_gate_then_exec_cwd_and_scrub(self):
        gate = _exec.run_cmd(self.root, "echo hi", allow=["echo"])
        self.assertEqual(gate["status"], "confirmation_required")
        os.environ["VENICE_API_KEY"] = "test-fake-key"
        try:
            r = _exec.run_cmd(
                self.root, "pwd; echo key=[${VENICE_API_KEY:-EMPTY}]", confirm=True)
        finally:
            os.environ.pop("VENICE_API_KEY", None)
        self.assertEqual(r["exit_code"], 0)
        self.assertIn(self.root, r["stdout"])       # cwd forced to root
        self.assertIn("key=[EMPTY]", r["stdout"])   # Venice key scrubbed from child

    def test_allowlisted_command_runs_after_confirm(self):
        r = _exec.run_cmd(self.root, "echo hola", allow=["echo"], confirm=True)
        self.assertEqual(r["status"], "ok")
        self.assertIn("hola", r["stdout"])

    def test_empty_command_errors(self):
        r = _exec.run_cmd(self.root, "   ", confirm=True)
        self.assertEqual(r["status"], "error")

    def test_timeout(self):
        r = _exec.run_cmd(self.root, "sleep 5", exec_timeout=1, confirm=True)
        self.assertEqual(r["status"], "error")
        self.assertIn("timed out", r["message"])


class TestGitCmd(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.root, ignore_errors=True)

    def test_duplicate_subcommand_in_args_rejected_before_exec(self):
        # #69: passing the subcommand inside args (mirroring a CLI invocation)
        # would run `git remote remote -v`; reject it with a tool error and never
        # reach subprocess.
        with mock.patch("venice.commands._exec.subprocess.run") as run:
            r = _exec.git_cmd(self.root, "remote", args=["remote", "-v"])
        self.assertEqual(r["status"], "error")
        self.assertIn("don't repeat the subcommand", r["message"])
        run.assert_not_called()

    def test_validated_args_reach_hardened_git(self):
        with mock.patch("venice.commands._exec.subprocess.run") as run:
            run.return_value = mock.Mock(returncode=0, stdout=" M source.py\n", stderr="")
            r = _exec.git_cmd(
                self.root, "status", args=["--short", "--", "source.py"],
                path_guard=lambda path: f"safe/{path}",
            )
        self.assertEqual(r["status"], "ok")
        argv = run.call_args[0][0]
        self.assertEqual(argv[0:2], ["git", "--no-pager"])
        self.assertIn("core.fsmonitor=false", argv)
        self.assertEqual(argv[-4:], ["status", "--short", "--", "safe/source.py"])

    def test_merge_base_is_allowed(self):
        # #80 part 1a: `venice review` pins its diff range to merge-base(base, HEAD),
        # so the subcommand has to reach subprocess. It computes a fork point and
        # writes nothing, which is why it belongs in the read-only set.
        with mock.patch("venice.commands._exec.subprocess.run") as run:
            run.return_value = mock.Mock(returncode=0, stdout="abc123\n", stderr="")
            r = _exec.git_cmd(self.root, "merge-base", args=["origin/master", "HEAD"])
        self.assertEqual(r["status"], "ok")
        self.assertEqual(run.call_args[0][0][-3:],
                         ["merge-base", "origin/master", "HEAD"])

    def test_mutating_subcommands_still_refused(self):
        # Anti-vacuity for the row above: widening the allow-list must not have
        # widened it to writes. Never reaches subprocess.
        for sub in ("commit", "checkout", "push", "merge", "reset", "clean"):
            with self.subTest(sub=sub):
                with mock.patch("venice.commands._exec.subprocess.run") as run:
                    r = _exec.git_cmd(self.root, sub)
                self.assertEqual(r["status"], "error")
                self.assertIn("read-only operations", r["message"])
                run.assert_not_called()

    def test_argument_level_mutations_and_escape_options_are_refused(self):
        cases = (
            ("remote", ["add", "audit", "https://example.invalid/repo"]),
            ("branch", ["-D", "work"]),
            ("branch", ["-f", "work", "HEAD"]),
            ("diff", ["--no-index", "--", "one", "two"]),
            ("diff", ["--output=leak.txt"]),
            ("diff", ["--ext-diff"]),
            ("diff", ["--textconv"]),
            ("blame", ["--contents", "outside", "--", "inside"]),
            ("rev-parse", ["--git-path", "config"]),
            ("rev-parse", ["README.md"]),
            ("show", ["HEAD:credentials"]),
            ("show", ["HEAD"]),
            ("diff", ["HEAD"]),
        )
        for sub, args in cases:
            with self.subTest(sub=sub, args=args):
                with mock.patch("venice.commands._exec.subprocess.run") as run:
                    result = _exec.git_cmd(
                        self.root, sub, args=list(args), path_guard=lambda path: path
                    )
                self.assertEqual(result["status"], "error")
                run.assert_not_called()

    def test_paths_require_authority_and_reject_magic(self):
        for args in (["--", "file.txt"], ["--", ":(top)file"], ["--", "*.py"]):
            with self.subTest(args=args):
                with mock.patch("venice.commands._exec.subprocess.run") as run:
                    result = _exec.git_cmd(self.root, "diff", args=args)
                self.assertEqual(result["status"], "error")
                run.assert_not_called()

    def test_git_environment_discards_all_inherited_git_controls(self):
        os.environ["GIT_DIR"] = "/outside"
        os.environ["GIT_EXTERNAL_DIFF"] = "/outside/helper"
        os.environ["VENICE_API_KEY"] = "test-fake-key"
        try:
            with mock.patch("venice.commands._exec.subprocess.run") as run:
                run.return_value = mock.Mock(returncode=0, stdout="", stderr="")
                result = _exec.git_cmd(self.root, "status")
        finally:
            os.environ.pop("GIT_DIR", None)
            os.environ.pop("GIT_EXTERNAL_DIFF", None)
            os.environ.pop("VENICE_API_KEY", None)
        self.assertEqual(result["status"], "ok")
        env = run.call_args.kwargs["env"]
        self.assertNotIn("GIT_DIR", env)
        self.assertNotIn("GIT_EXTERNAL_DIFF", env)
        self.assertNotIn("VENICE_API_KEY", env)
        self.assertEqual(env["GIT_CONFIG_NOSYSTEM"], "1")
        self.assertEqual(env["GIT_CONFIG_GLOBAL"], os.devnull)
        self.assertEqual(env["GIT_OPTIONAL_LOCKS"], "0")


if __name__ == "__main__":
    unittest.main()
