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
import stat
import tempfile
import unittest
from pathlib import Path

import venice
from tests import _drive
from tests import _venice_fake_server as fake

_HAS_OPENAI = importlib.util.find_spec("openai") is not None

needs_pty = unittest.skipUnless(_drive.HAS_PEXPECT, _drive.SKIP_REASON)
needs_openai = unittest.skipUnless(_HAS_OPENAI, "openai SDK not installed")


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


# --------------------------------------------------------------------------
# C. the charge-confirmation gate (_shared.confirm_or_exit)
# --------------------------------------------------------------------------

@needs_pty
class TestDriveCharge(_DriveCase):
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


# --------------------------------------------------------------------------
# C'. the same gates with no pty at all
#
# The `not sys.stdin.isatty()` arms are unreachable through a pty by definition,
# so these drive the CLI over plain pipes instead. Deliberately NOT gated on
# pexpect: they are real coverage and should keep running for a contributor who
# hasn't installed the [test] extra.
# --------------------------------------------------------------------------

class TestDriveNonInteractive(_DriveCase):
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


# --------------------------------------------------------------------------
# D. login -- the one flow that requires a real terminal
# --------------------------------------------------------------------------

@needs_pty
class TestDriveLogin(_DriveCase):
    server = False

    def test_login_writes_sandboxed_creds_and_never_echoes_the_key(self):
        # Obviously fake, and sandboxed to a tmp HOME (CLAUDE.md): the real
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

    def test_repl_ctrl_d_exits_zero(self):
        with self.cli("chat", "-i") as d:
            d.expect("you> ")
            d.send_eof()
            self.assertEqual(d.wait(), 0)


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
