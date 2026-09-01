"""Drive-the-real-CLI tests (#80): spawn `venice` on a pty and assert on output.

Every other test in this suite reaches the CLI at a handler seam -- `chat._run`
with a hand-built Namespace, `builtins.input` patched with a side_effect list.
That proves the branch executes. It does not prove the prompt reaches a terminal,
in the right order, before the thing it is asking about. The escape this suite
exists to catch is "suite green, feature unusable interactively".

So these tests run the real `python -m venice` as a child process on a pty,
against the loopback fake API in `_venice_fake_server.py`, with `$HOME`
redirected to a tmpdir. Still hermetic: no network, no real key (see
`_drive.py`'s docstring and CONTRIBUTING.md).

Two of these are deliberately long, interleaved dialogues rather than one
assertion each -- `test_repl_streams_switches_model_resets_and_autosaves` and
`test_code_plan_gate_reprompts_edits_then_declines`. Interactive breakage lives
in the transitions between prompts, so a suite of one-shot cases would miss
exactly what this suite is for.

The pty tests skip without the `[test]` extra; the non-pty ones always run.
"""
import importlib.util
import json
import os
import shutil
import site
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import venice
from tests import _drive
from tests import _venice_fake_server as fake

_HAS_OPENAI = importlib.util.find_spec("openai") is not None

needs_pty = unittest.skipUnless(_drive.HAS_PEXPECT, _drive.SKIP_REASON)
needs_openai = unittest.skipUnless(_HAS_OPENAI, "openai SDK not installed")


class TestDrivePythonPath(unittest.TestCase):
    """The child's import path, and the two things that keep it coherent (#107)."""

    def test_user_site_follows_parent_effective_configuration(self):
        with tempfile.TemporaryDirectory() as user_site:
            with mock.patch.object(site, "getusersitepackages", return_value=user_site):
                with mock.patch.object(site, "ENABLE_USER_SITE", False):
                    disabled = _drive._python_path().split(os.pathsep)
                with mock.patch.object(site, "ENABLE_USER_SITE", True):
                    enabled = _drive._python_path().split(os.pathsep)

        self.assertNotIn(user_site, disabled)
        self.assertIn(user_site, enabled)

    def test_child_disables_home_derived_user_site(self):
        # Defense in depth, and deliberately so: the tmp $HOME has no `.local`,
        # so the child derives no user site today with or without this. It
        # pins the *intent* -- membership in the user site dir stays something
        # the parent states through PYTHONPATH, never something $HOME implies
        # the day a fixture starts writing into the redirected home.
        with tempfile.TemporaryDirectory() as home:
            self.assertEqual(_drive.build_env(home)["PYTHONNOUSERSITE"], "1")


class _DriveCase(unittest.TestCase):
    """Fresh tmp HOME and project dir per test; one fake API per class."""

    server = True

    @classmethod
    def setUpClass(cls):
        cls.api = fake.FakeVenice().start() if cls.server else None

    @classmethod
    def tearDownClass(cls):
        if cls.api is not None:
            cls.api.stop()

    def setUp(self):
        if self.api is not None:
            self.api.reset()
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        root = Path(self._tmp.name)
        self.home = root / "home"
        self.project = root / "project"
        self.home.mkdir()
        self.project.mkdir()

    @property
    def base_url(self):
        return self.api.base_url if self.api is not None else None

    def cli(self, *argv, **kw):
        kw.setdefault("home", self.home)
        kw.setdefault("base_url", self.base_url)
        kw.setdefault("cwd", self.project)
        return _drive.cli(*argv, **kw)

    def run_cli(self, *argv, **kw):
        kw.setdefault("home", self.home)
        kw.setdefault("base_url", self.base_url)
        kw.setdefault("cwd", self.project)
        return _drive.run(*argv, **kw)

    def sessions(self):
        return sorted((self.home / ".config" / "venice" / "sessions").glob("*.json"))


# --------------------------------------------------------------------------
# A. smoke -- isolates harness breakage from CLI breakage
# --------------------------------------------------------------------------

@needs_pty
class TestDriveSmoke(_DriveCase):
    server = False

    def test_version_prints_and_exits_zero(self):
        with self.cli("--version") as d:
            d.expect("venice %s" % venice.__version__)
            self.assertEqual(d.wait(), 0)

    def test_bare_venice_prints_help_and_exits_2(self):
        with self.cli() as d:
            d.expect("usage: venice")
            d.expect("COMMAND")
            self.assertEqual(d.wait(), 2)


# --------------------------------------------------------------------------
# B. the HTTP path end to end, in a real process
# --------------------------------------------------------------------------

@needs_pty
class TestDriveHttp(_DriveCase):
    def test_balance_prints_total_from_fake_server(self):
        # $VENICE_BASE_URL -> build_client_from_auth -> urllib -> fetch_balance,
        # none of it mocked. 12.5 USD + 0.0 DIEM, via billing.format_usd.
        with self.cli("balance") as d:
            d.expect("$12.50 USD")
            self.assertEqual(d.wait(), 0)
        self.assertEqual(self.api.paths, ["/api_keys/rate_limits"])
        self.assertTrue(self.api.requests[0]["has_auth"])

    @needs_openai
    def test_plain_non_stream_chat_reports_usage_on_stderr(self):
        self.api.reply("HELLO-FROM-FAKE")
        cp = self.run_cli("chat", "Say something", "--no-stream")
        self.assertEqual(cp.returncode, 0, cp.stderr)
        self.assertEqual(cp.stdout, "HELLO-FROM-FAKE\n")
        self.assertEqual(
            cp.stderr,
            "usage: prompt=11 completion=3 total=14\n",
        )
        self.assertEqual(self.api.paths, ["/models", "/chat/completions"])

    @needs_openai
    def test_plain_streaming_chat_ctrl_c_reports_partial_output(self):
        release = self.api.reply_paused_stream("PARTIAL-FROM-FAKE")
        self.addCleanup(release.set)

        with self.cli("chat", "Say something") as d:
            # Seeing the first delta proves the child is inside the plain streaming
            # handler before SIGINT is delivered; the fake holds the SSE connection
            # open, so this synchronization has no sleep or timing race.
            d.expect("PARTIAL-FROM-FAKE")
            d.ctrl_c()
            d.expect("chat: aborted (partial output may appear above)")
            self.assertEqual(d.wait(), 130)
            self.assertNotIn("Traceback", d.transcript)

        release.set()
        self.assertEqual(self.api.paths, ["/models", "/chat/completions"])


# --------------------------------------------------------------------------
# C. the charge-confirmation gate (_shared.confirm_or_exit)
# --------------------------------------------------------------------------

@needs_pty
class TestDriveCharge(_DriveCase):
    def test_cache_probe_confirms_before_two_paid_calls(self):
        self.api.reply_usage(
            {"cached_tokens": 0, "cache_creation_input_tokens": 8},
            prompt_tokens=12,
        )
        self.api.reply_usage(
            {"cached_tokens": 8, "provider_extension": "preserved"},
            prompt_tokens=12,
        )

        with self.cli(
            "cache-probe", "--model", "llama-3.3-70b",
            "--prefix-tokens", "8",
        ) as d:
            d.expect("Estimated maximum cost:")
            d.expect("Proceed? [y/N] ")
            # The free catalog lookup happened, but no paid completion can precede
            # the operator's answer.
            self.assertEqual(self.api.paths, ["/models"])
            d.send("y")
            d.expect(
                'prompt_tokens_details={"cached_tokens":0,'
                '"cache_creation_input_tokens":8}'
            )
            d.expect(
                'prompt_tokens_details={"cached_tokens":8,'
                '"provider_extension":"preserved"}'
            )
            d.expect("llama-3.3-70b: warms=8")
            self.assertEqual(d.wait(), 0)

        self.assertEqual(
            self.api.paths,
            ["/models", "/chat/completions", "/chat/completions"],
        )
        bodies = self.api.bodies("/chat/completions")
        self.assertEqual(len(bodies), 2)
        self.assertEqual(bodies[0], bodies[1])
        self.assertEqual(bodies[0]["max_tokens"], 1)
        self.assertFalse(bodies[0]["stream"])
        self.assertTrue(
            bodies[0]["messages"][0]["content"].startswith(
                "CACHE-PROBE-V1 SIZE=8\n"
            )
        )

    def test_image_confirm_accept_writes_the_file(self):
        with self.cli("image", "a cat", "--name", "drive-test") as d:
            d.expect("Estimated cost: $0.0100 USD (1 image, model=venice-sd35)")
            d.expect("Balance:")
            d.expect("Proceed? [y/N] ")
            d.send("y")
            d.expect("wrote %d bytes to" % len(fake.PNG_BYTES))
            self.assertEqual(d.wait(), 0)

        out = self.project / "drive-test.png"
        self.assertEqual(out.read_bytes(), fake.PNG_BYTES)
        self.assertEqual(
            self.api.paths,
            ["/models", "/api_keys/rate_limits", "/image/generate"],
        )

    def test_config_defaults_reach_the_confirm_gate(self):
        """#57 Class C1: `defaults.image.variants` is a COST MULTIPLIER, so it has
        to be visible in the very line the user is being asked to approve. Only a
        real pty run through the real parser proves the config value survives
        `apply_defaults` -> `apply_literals` and lands in the quote.

        The model stays at the built-in id on purpose: the fake catalog is keyed
        to `image.DEFAULT_IMAGE_MODEL`, and an unknown id would make the price
        lookup fail and change the shape of this line.
        """
        cfgdir = self.home / ".config" / "venice"
        cfgdir.mkdir(parents=True, exist_ok=True)
        (cfgdir / "config.json").write_text(json.dumps({
            "version": 1,
            "mcpServers": {},
            "defaults": {"image": {"model": "venice-sd35", "variants": 2}},
        }), encoding="utf-8")

        with self.cli("image", "a cat", "--name", "drive-cfg") as d:
            d.expect("Estimated cost: $0.0200 USD (2 images, model=venice-sd35)")
            d.expect("Proceed? [y/N] ")
            d.send("y")
            d.expect("wrote %d bytes to" % len(fake.PNG_BYTES))
            self.assertEqual(d.wait(), 0)

        self.assertEqual(len(sorted(self.project.glob("drive-cfg*.png"))), 2)

    def test_image_confirm_decline_makes_no_paid_call(self):
        with self.cli("image", "a cat", "--name", "drive-test") as d:
            d.expect("Proceed? [y/N] ")
            d.send("n")
            d.expect("aborted by user")
            self.assertEqual(d.wait(), 1)

        # The point of the gate: declining must cost nothing. Only an in-process
        # fake can prove the paid endpoint was never reached.
        self.assertNotIn("/image/generate", self.api.paths)
        self.assertEqual(list(self.project.glob("*.png")), [])

    @needs_openai
    def test_agent_tool_call_confirm_accept_reaches_the_second_turn(self):
        """#114: drive the real agent loop through a paid tool and back."""
        self.api.reply_tool_call("venice_image", {"prompt": "a tiny cat"})
        self.api.reply("TOOL-LOOP-COMPLETE")

        with self.cli(
            "chat", "make an image", "--tools", "--tool", "venice_image",
            "--max-spend", "0", "--json",
        ) as d:
            d.expect("image: estimated cost $0.0100 is over the auto-approve cap")
            d.expect("Proceed? [y]es / [a]ll (accept rest) / [N]o ")
            d.send("y")
            d.expect('"final": "TOOL-LOOP-COMPLETE"')
            d.expect('"prompt_tokens": 22')
            d.expect('"completion_tokens": 6')
            d.expect('"turns": 1')
            self.assertEqual(d.wait(), 0)

        outputs = list(self.project.glob("venice-image-*.png"))
        self.assertEqual(len(outputs), 1)
        self.assertEqual(outputs[0].read_bytes(), fake.PNG_BYTES)
        self.assertEqual(self.api.paths.count("/image/generate"), 1)

        chat_bodies = self.api.bodies("/chat/completions")
        self.assertEqual(len(chat_bodies), 2)
        assistant, result = chat_bodies[1]["messages"][-2:]
        self.assertEqual(assistant["role"], "assistant")
        self.assertEqual(assistant["tool_calls"][0]["id"], "call_fake_0")
        self.assertEqual(assistant["tool_calls"][0]["function"]["name"],
                         "venice_image")
        self.assertEqual(result["role"], "tool")
        self.assertEqual(result["tool_call_id"], "call_fake_0")
        self.assertEqual(result["name"], "venice_image")
        self.assertEqual(json.loads(result["content"])["status"], "ok")


# --------------------------------------------------------------------------
# C'. the same gates with no pty at all
#
# The `not sys.stdin.isatty()` arms are unreachable through a pty by definition,
# so these drive the CLI over plain pipes instead. Deliberately NOT gated on
# pexpect: they are real coverage and should keep running for a contributor who
# hasn't installed the [test] extra.
# --------------------------------------------------------------------------

class TestDriveNonInteractive(_DriveCase):
    def test_broken_stdout_pipe_is_quiet_exit_141(self):
        # More than a pipe buffer makes the EPIPE deterministic: `head` exits
        # after its first line while the real CLI still has output left to write.
        self.api.set_models("all", [
            {"id": "pipe-model-%05d" % i, "type": "text"}
            for i in range(10_000)
        ])
        producer = subprocess.Popen(
            [sys.executable, "-m", "venice", "models", "--type", "all"],
            env=_drive.build_env(self.home, base_url=self.base_url),
            cwd=str(self.project),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.addCleanup(lambda: producer.kill() if producer.poll() is None else None)
        self.assertIsNotNone(producer.stdout)
        self.assertIsNotNone(producer.stderr)
        head = subprocess.run(
            ["head", "-1"],
            stdin=producer.stdout,
            capture_output=True,
            text=True,
            timeout=_drive.DEFAULT_TIMEOUT,
        )
        producer.stdout.close()
        self.assertEqual(producer.wait(timeout=_drive.DEFAULT_TIMEOUT), 141)
        producer_err = producer.stderr.read()
        producer.stderr.close()

        self.assertEqual(head.returncode, 0, head.stderr)
        self.assertEqual(head.stdout, "### text (10000)\n")
        self.assertEqual(producer_err, "")

    def test_image_non_interactive_requires_yes(self):
        cp = self.run_cli("image", "a cat", "--name", "drive-test")
        self.assertEqual(cp.returncode, 1)
        self.assertIn("non-interactive; pass --yes to confirm the charge.", cp.stderr)
        self.assertNotIn("/image/generate", self.api.paths)
        self.assertEqual(list(self.project.glob("*.png")), [])

    @needs_openai
    def test_code_non_interactive_refuses_without_auto(self):
        cp = self.run_cli("code", "add a docstring", "--root", str(self.project))
        self.assertEqual(cp.returncode, 2)
        self.assertIn("code: refusing to run unattended without --auto", cp.stderr)
        # Encodes the intent of the comment at code.py:696 -- it must abort
        # *before* spending a plan turn.
        self.assertNotIn("/chat/completions", self.api.paths)

    @needs_openai
    def test_code_footer_reports_the_cache_state(self):
        # #100 at the process level. The run footer had NO pty/subprocess coverage
        # at all before this -- and it is the single surface a one-shot `venice code`
        # puts a cache collapse in front of an operator, so "it renders in a unit
        # test" was never the claim that mattered. The fake's usage block carries no
        # `prompt_tokens_details` (a live glm/kimi response's shape), so the honest
        # answer is n/a, not the 0.0% that hid a real regression for three days.
        self.api.reply("PLAN: add a docstring")
        self.api.reply("done")
        self.api.reply("ACCEPTANCE: PASS")
        cp = self.run_cli("code", "add a docstring", "--root", str(self.project),
                          "--auto")
        self.assertEqual(cp.returncode, 0, cp.stderr)
        footer = [ln for ln in cp.stderr.splitlines()
                  if ln.startswith("code: ") and "wall" in ln]
        self.assertEqual(len(footer), 1, cp.stderr)
        self.assertTrue(footer[0].endswith("cache n/a"), footer[0])
        self.assertNotIn("0.0% hit", cp.stderr)
        # ...and the machine-readable half agrees, through the same real process.
        usage = json.loads(self.sessions()[0].read_text(encoding="utf-8"))["usage"]
        self.assertIsNone(usage["cache_hit_percent"])


# --------------------------------------------------------------------------
# C2. venice review (#80 part 1a) -- the real CLI over a real git repo.
#
# The unit suite mocks `_agent.run_review`, so nothing there proves the
# subcommand actually collects a diff from git, reaches the API, and comes back
# with an exit code. These do, against the loopback fake.
# --------------------------------------------------------------------------

class TestDriveReview(_DriveCase):
    """`venice review` end to end: real git, real process, fake API."""

    def _repo(self, extra=None):
        """A project dir that is a git repo with one committed baseline."""
        env = {"GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_SYSTEM": "/dev/null",
               "GIT_AUTHOR_NAME": "T", "GIT_AUTHOR_EMAIL": "t@example.invalid",
               "GIT_COMMITTER_NAME": "T", "GIT_COMMITTER_EMAIL": "t@example.invalid"}
        full = dict(os.environ)
        full.update(env)

        def git(*a):
            subprocess.run(["git", *a], cwd=str(self.project), env=full,
                           check=True, capture_output=True, text=True)

        git("init", "-q", "-b", "master", ".")
        (self.project / "app.py").write_text("def add(a, b):\n    return a + b\n")
        git("add", "-A")
        git("commit", "-qm", "baseline")
        for rel, text in (extra or {}).items():
            p = self.project / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text)

    @unittest.skipUnless(shutil.which("git"), "git is not installed")
    @needs_openai
    def test_review_collects_a_real_diff_and_reports_findings(self):
        self._repo({"app.py": "def add(a, b):\n    return a - b\n"})
        self.api.reply(
            "SCOPE: app.py\n\nFINDINGS:\napp.py:2 [blocker] add() subtracts\n"
            "WHY: the operator is wrong.\n"
            "REPRO: add(2, 1) returns 1 instead of 3.\n"
            "FIX: restore `a + b`.\n\nNOT CHECKED: nothing\n\nREVIEW: FINDINGS"
        )
        # The default is 2 rounds, so queue the second pass too: it finds nothing
        # new, which is what makes the until-dry loop stop at 2 instead of 3.
        self.api.reply("SCOPE: app.py\n\nFINDINGS: none\n\nNOT CHECKED: nothing\n\n"
                       "REVIEW: FINDINGS")
        cp = self.run_cli("review", "--root", str(self.project), "--json")
        self.assertEqual(cp.returncode, 1, cp.stderr)      # blocker >= --fail-on major
        env = json.loads(cp.stdout)
        self.assertEqual(env["verdict"], "findings")
        self.assertEqual(env["files_reviewed"], ["app.py"])
        self.assertEqual(env["findings"][0]["severity"], "blocker")
        self.assertEqual(env["findings"][0]["line"], 2)
        self.assertRegex(env["base_sha"], r"^[0-9a-f]{40}$")
        # The fake ships two function-calling models from different families, so
        # the reviewer must NOT be the catalog default the author would use.
        self.assertEqual(env["model"], "venice-uncensored")
        self.assertTrue(env["decorrelated"])
        # #80's separation constraint, at the process level: a real run that found a
        # real blocker still leaves nothing behind.
        self.assertEqual(list(self.project.glob("*.json")), [])
        self.assertFalse((self.project / ".venice").exists())
        self.assertEqual(self.sessions(), [])

    @unittest.skipUnless(shutil.which("git"), "git is not installed")
    def test_docs_only_review_spends_nothing(self):
        # The cost-discipline claim as a process-level fact: no /chat/completions,
        # and no /models either -- triage runs before the client is ever built.
        self._repo({"README.md": "hello\nmore\n"})
        cp = self.run_cli("review", "--root", str(self.project), "--json")
        self.assertEqual(cp.returncode, 0, cp.stderr)
        env = json.loads(cp.stdout)
        self.assertEqual(env["status"], "skipped")
        self.assertIsNone(env["verdict"])
        self.assertNotIn("/chat/completions", self.api.paths)
        self.assertEqual(self.api.paths, [])

    @unittest.skipUnless(shutil.which("git"), "git is not installed")
    def test_review_outside_a_git_repo_exits_2(self):
        cp = self.run_cli("review", "--root", str(self.project))
        self.assertEqual(cp.returncode, 2)
        self.assertIn("not a git repository", cp.stderr)


# --------------------------------------------------------------------------
# D. login -- the one flow that requires a real terminal
# --------------------------------------------------------------------------

@needs_pty
class TestDriveLogin(_DriveCase):
    server = False

    def test_login_writes_sandboxed_creds_and_never_echoes_the_key(self):
        # Obviously fake, and sandboxed to a tmp HOME (AGENTS.md): the real
        # credentials file is never touched or read.
        typed = "test-fake-key-0123456789"
        with self.cli("login", api_key=None) as d:
            d.expect("Paste your Venice API key")
            d.expect("API key: ")
            # send_secret, not send: a failing expect below prints the
            # transcript, and a plain send would put the typed value in it.
            d.send_secret(typed)
            d.expect("Saved %d-char key to" % len(typed))
            d.expect("(mode 0600).")
            self.assertEqual(d.wait(), 0)

        creds = self.home / ".config" / "venice" / "credentials"
        self.assertTrue(creds.exists())
        self.assertEqual(stat.S_IMODE(creds.stat().st_mode), 0o600)
        self.assertEqual(stat.S_IMODE(creds.parent.stat().st_mode), 0o700)

        # The regression that matters: getpass silently degrading to a cleartext
        # input(). A mocked getpass can never prove the *terminal* stayed quiet.
        # `screen` is reads-only -- `transcript` would contain our own send.
        self.assertNotIn(typed, d.screen)
        # ...and the failure path can't leak it either, whatever goes red.
        self.assertNotIn(typed, d.transcript)

    def test_login_ctrl_c_exits_130(self):
        with self.cli("login", api_key=None) as d:
            d.expect("API key: ")
            # getpass clears ECHO but leaves ISIG, so the pty INTR character
            # really does deliver SIGINT to the child.
            d.ctrl_c()
            d.expect("aborted.")
            self.assertEqual(d.wait(), 130)

        self.assertFalse((self.home / ".config" / "venice" / "credentials").exists())


# --------------------------------------------------------------------------
# E. interleaved flow #1 -- the chat REPL
# --------------------------------------------------------------------------

@needs_pty
@needs_openai
class TestDriveChatRepl(_DriveCase):
    def test_repl_streams_switches_model_resets_and_autosaves(self):
        self.api.reply("HELLO-FROM-FAKE")
        self.api.reply("SECOND-REPLY")

        with self.cli("chat", "-i") as d:
            d.expect("venice chat -- interactive (model llama-3.3-70b")
            d.expect("/exit or Ctrl-D to quit.")
            d.expect("you> ")

            # Turn 1: a real SSE round-trip reaching the terminal.
            d.send("Say hi")
            d.expect("HELLO-FROM-FAKE")
            d.expect("usage: prompt=11 completion=3 total=14")
            d.expect("you> ")

            # A rejected /model must not kill the session -- it re-prompts.
            d.send("/model no-such-model")
            d.expect("chat: unknown text model 'no-such-model'")
            d.expect("available: llama-3.3-70b, venice-uncensored")
            d.expect("you> ")

            d.send("/model venice-uncensored")
            d.expect("(model -> venice-uncensored)")
            d.expect("you> ")

            # Turn 2: proves the switch reached the wire, not just a state dict.
            d.send("Again")
            d.expect("SECOND-REPLY")
            d.expect("you> ")

            d.send("/nope")
            d.expect("unknown command /nope; /help for the list")
            d.expect("you> ")

            d.send("/reset")
            d.expect("(conversation cleared)")
            d.expect("you> ")

            d.send("/exit")
            self.assertEqual(d.wait(), 0)

        bodies = self.api.bodies("/chat/completions")
        self.assertEqual(len(bodies), 2)
        self.assertEqual(bodies[0]["model"], "llama-3.3-70b")
        self.assertEqual(bodies[1]["model"], "venice-uncensored")
        self.assertEqual(bodies[1]["messages"][-1]["content"], "Again")

        # The autosave really wrote an envelope into the sandboxed HOME, and it
        # reflects the end state of the dialogue rather than its start.
        saved = self.sessions()
        self.assertEqual(len(saved), 1)
        envelope = json.loads(saved[0].read_text(encoding="utf-8"))
        # The /model switch survived all the way to disk...
        self.assertEqual(envelope["model"], "venice-uncensored")
        # ...both turns were accounted for (11 prompt tokens each)...
        self.assertEqual(envelope["usage"]["prompt_tokens"], 22)
        # ...and /reset really cleared the transcript, not just the display.
        self.assertEqual(envelope["messages"], [])

    def test_repl_persists_a_per_call_trace(self):
        # #99 through the REAL CLI on a real pty. Each queued reply is its own
        # `create()`, so two turns must leave two rows -- and because the fake's usage
        # block carries no `prompt_tokens_details` (#114), every row's cache field must
        # persist as null rather than a confident 0.
        self.api.reply("FIRST-REPLY")
        self.api.reply("SECOND-REPLY")

        with self.cli("chat", "-i") as d:
            d.expect("you> ")
            d.send("One")
            d.expect("FIRST-REPLY")
            d.expect("you> ")
            d.send("Two")
            d.expect("SECOND-REPLY")
            d.expect("you> ")
            d.send("/exit")
            self.assertEqual(d.wait(), 0)

        envelope = json.loads(self.sessions()[0].read_text(encoding="utf-8"))
        usage = envelope["usage"]
        self.assertEqual(usage["prompt_tokens"], 22)   # the existing aggregate...
        self.assertEqual(usage["api_calls_total"], 2)  # ...now itemized
        rows = usage["api_calls"]
        self.assertEqual([r["n"] for r in rows], [1, 2])
        self.assertEqual([r["prompt_tokens"] for r in rows], [11, 11])
        self.assertEqual([r["cache_read_tokens"] for r in rows], [None, None])
        for r in rows:
            # The streamed path brackets its own window through the real process.
            self.assertIsNotNone(r["seconds"])
        self.assertEqual(usage["context_events"], [])
        self.assertEqual(usage["buckets"], {})  # #101: nothing off-loop happened

    def test_repl_records_an_auto_compaction_event(self):
        # #99's other half, driven end to end. `--compact-threshold 1` makes
        # `Budget.over` short-circuit on the fake's observed 11 prompt tokens, so the
        # second turn compacts -- and the event lands in the envelope with the
        # server-reported number, which is the pre-reset read the ordering depends on.
        # THREE turns, not two: at the top of turn 2 the history is a single group,
        # so `split_for_compaction` leaves an empty prefix and declines. Turn 3 is the
        # first checkpoint with something to summarize -- hence the 4th queued reply,
        # which is consumed by the summarization call rather than shown.
        self.api.reply("FIRST-REPLY")
        self.api.reply("SECOND-REPLY")
        self.api.reply("SUMMARY-OF-EARLIER")
        self.api.reply("THIRD-REPLY")

        with self.cli("chat", "-i", "--auto-compact",
                      "--compact-threshold", "1", "--compact-keep-turns", "1") as d:
            d.expect("you> ")
            d.send("One")
            d.expect("FIRST-REPLY")
            d.expect("you> ")
            d.send("Two")
            d.expect("SECOND-REPLY")
            d.expect("you> ")
            d.send("Three")
            d.expect("auto-compacted history")
            d.expect("THIRD-REPLY")
            d.expect("you> ")
            d.send("/exit")
            self.assertEqual(d.wait(), 0)

        usage = json.loads(self.sessions()[0].read_text(encoding="utf-8"))["usage"]
        events = usage["context_events"]
        self.assertEqual(len(events), 1)
        ev = events[0]
        self.assertEqual(ev["kind"], "compaction")
        self.assertEqual(ev["trigger"], "auto")
        self.assertEqual(ev["observed_tokens_before"], 11)
        self.assertGreater(ev["messages_before"], ev["messages_after"])
        # #101: the summarization call is billed, and this tier is the only one that
        # proves it survives the real `_autosave` -> `to_dict` -> JSON round trip.
        self.assertIn("cost", ev)
        self.assertEqual(usage["buckets"]["compaction"]["calls"], 1)
        self.assertGreater(usage["buckets"]["compaction"]["prompt_tokens"], 0)
        # ...and stays OUT of the main-loop trace, which is the whole isolation claim.
        self.assertEqual(usage["api_calls_total"], len(usage["api_calls"]))
        self.assertNotIn("compaction", str(usage["api_calls"]))

    def test_sessions_show_stays_bounded_through_the_real_cli(self):
        # The `sessions show` line is a one-line repr of the whole usage dict; #99
        # bounds it. Driven through the real process so the fix is proven where an
        # operator actually meets it.
        self.api.reply("HELLO-FROM-FAKE")

        with self.cli("chat", "-i") as d:
            d.expect("you> ")
            d.send("Say hi")
            d.expect("HELLO-FROM-FAKE")
            d.expect("you> ")
            d.send("/exit")
            self.assertEqual(d.wait(), 0)

        # `latest` resolves the most recent CODE session, and this is a chat one --
        # target the id the autosave actually wrote.
        sid = json.loads(self.sessions()[0].read_text(encoding="utf-8"))["id"]
        with self.cli("sessions", "show", sid) as d:
            d.expect("api_calls: 1 row")
            self.assertEqual(d.wait(), 0)

    def test_repl_usage_reports_unknown_cache_state(self):
        # #98 through the REAL CLI on a real pty: the fake server's usage block
        # carries no `prompt_tokens_details` (exactly like a live glm/kimi response,
        # and like the spec's own nullable example), so /usage must say it does not
        # know rather than print a "0.0%" that reads as a measurement.
        self.api.reply("HELLO-FROM-FAKE")

        with self.cli("chat", "-i") as d:
            d.expect("you> ")
            d.send("Say hi")
            d.expect("HELLO-FROM-FAKE")
            d.expect("you> ")

            d.send("/usage")
            d.expect("session usage:")
            d.expect("cache breakdown not reported")
            d.expect("cache hit rate: n/a")
            d.expect("you> ")

            # #100: /cost gained the same claim, and must agree with /usage. An
            # operator runs these seconds apart, so a disagreement here reads as
            # two different states rather than one state phrased twice.
            d.send("/cost")
            d.expect("cache n/a")
            d.expect("you> ")

            d.send("/exit")
            self.assertEqual(d.wait(), 0)

    def test_repl_ctrl_d_exits_zero(self):
        with self.cli("chat", "-i") as d:
            d.expect("you> ")
            d.send_eof()
            self.assertEqual(d.wait(), 0)

    def test_repl_ctrl_d_flushes_a_slash_only_session(self):
        # #92: /model commits no turn, so the ^D exit is the only flush there is.
        # The unit tests pin the `finally`; this pins that it survives a real pty.
        with self.cli("chat", "-i") as d:
            d.expect("you> ")
            d.send("/model venice-uncensored")
            d.expect("(model -> venice-uncensored)")
            d.expect("you> ")
            d.send_eof()
            self.assertEqual(d.wait(), 0)
            self.assertNotIn("Traceback", d.transcript)

        saved = self.sessions()
        self.assertEqual(len(saved), 1)
        envelope = json.loads(saved[0].read_text(encoding="utf-8"))
        self.assertEqual(envelope["model"], "venice-uncensored")

    def test_repl_ctrl_c_at_prompt_reprompts_and_keeps_the_session(self):
        # #92: `Driver.ctrl_c` had exactly one call site repo-wide (venice login) --
        # the REPL's own signal behaviour was only ever covered by mocked input().
        # A real SIGINT at the prompt must discard the line and come back, not exit.
        self.api.reply("ALIVE-AFTER-INTERRUPT")
        with self.cli("chat", "-i") as d:
            d.expect("you> ")
            d.send_partial("HALF-TYPED-DISCARDED")
            # Deliberately matching our own echo, which the harness rules otherwise
            # warn against: it is the sync point. Firing the interrupt before
            # readline has taken the keystrokes makes the test a coin flip.
            d.expect("HALF-TYPED-DISCARDED")
            d.ctrl_c()
            d.expect("you> ")
            d.send("Say something")
            d.expect("ALIVE-AFTER-INTERRUPT")
            d.expect("you> ")
            d.send("/exit")
            self.assertEqual(d.wait(), 0)
            self.assertNotIn("Traceback", d.transcript)

        # The interrupted line never became a turn.
        bodies = self.api.bodies("/chat/completions")
        self.assertEqual(len(bodies), 1)
        self.assertEqual(bodies[0]["messages"][-1]["content"], "Say something")


# --------------------------------------------------------------------------
# F. interleaved flow #2 -- the `venice code` plan-acceptance gate
# --------------------------------------------------------------------------

@needs_pty
@needs_openai
class TestDriveCodePlanGate(_DriveCase):
    def test_code_plan_gate_reprompts_edits_then_declines(self):
        self.api.reply("PLAN-ONE: add a docstring")
        self.api.reply("PLAN-TWO: a shorter plan")

        with self.cli("code", "add a docstring", "--root", str(self.project)) as d:
            d.expect("PLAN-ONE")
            d.expect("Accept and run? ")

            # Invalid input must re-prompt, not abort.
            d.send("x")
            d.expect("Please answer a, s, e, or n.")
            d.expect("Accept and run? ")

            # A nested second prompt, then a live round-trip mid-dialogue.
            d.send("e")
            d.expect("Describe the change to the plan (blank to cancel): ")
            d.send("make it shorter")
            d.expect("PLAN-TWO")
            d.expect("Accept and run? ")

            # Bare Enter is the [N] default.
            d.send("")
            d.expect("code: plan not accepted; aborting")
            self.assertEqual(d.wait(), 1)

        # Two plan turns (the original and the revision), and nothing written.
        self.assertEqual(len(self.api.bodies("/chat/completions")), 2)
        self.assertEqual(list(self.project.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
