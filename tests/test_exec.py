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

    def test_duplicate_subcommand_in_args_rejected_before_exec(self):
        # #69: passing the subcommand inside args (mirroring a CLI invocation)
        # would run `git remote remote -v`; reject it with a tool error and never
        # reach subprocess.
        with mock.patch("venice.commands._exec.subprocess.run") as run:
            r = _exec.git_cmd(self.root, "remote", args=["remote", "-v"])
        self.assertEqual(r["status"], "error")
        self.assertIn("don't repeat the subcommand", r["message"])
        run.assert_not_called()

    def test_normal_args_are_unchanged(self):
        # Regression guard: args that do NOT repeat the subcommand still build the
        # expected argv and run.
        with mock.patch("venice.commands._exec.subprocess.run") as run:
            run.return_value = mock.Mock(returncode=0, stdout="origin\n", stderr="")
            r = _exec.git_cmd(self.root, "remote", args=["-v"])
        self.assertEqual(r["status"], "ok")
        self.assertEqual(run.call_args[0][0], ["git", "remote", "-v"])


if __name__ == "__main__":
    unittest.main()
