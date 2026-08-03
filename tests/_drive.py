"""Spawn harness for the #80 "drive the real CLI" tests.

The rest of the suite tests interactive flows by patching ``builtins.input``
with a ``side_effect`` list. That proves the *branch* runs; it cannot prove the
prompt reaches a terminal in a usable order. This module closes that gap: it
spawns the real ``python -m venice`` on a pty and lets tests assert on what a
person would actually see.

Hermeticity is preserved, by different means than the mock-based tests: a tmp
``$HOME`` (``config.HOME`` is resolved at import, so this sandboxes credentials,
config.json, secrets, sessions, memory and personas in one move), a
``VENICE_API_KEY=test-fake-key`` placeholder, and ``$VENICE_BASE_URL`` pointed
at the loopback fake in ``_venice_fake_server.py``. No network, no real key.

``pexpect`` is imported lazily inside :func:`spawn` so this module imports fine
on Python 3.9 and without the ``[test]`` extra -- same house rule as the openai
and mcp SDKs. Gate pty tests on :data:`HAS_PEXPECT`.

Rules for writing drive tests (each one is a flake source paid for once here):

* **Never match across a newline, and never anchor with ``^``/``$``.** A pty
  translates ``\\n`` to ``\\r\\n``.
* **No expected string may be a substring of something you just sent.** readline
  echoes typed input back onto the pty *before* the response arrives, and
  ``expect`` scans forward -- it would match your own keystrokes. Keep fake
  replies unlike anything you type (``HELLO-FROM-FAKE``, not ``hi``).
* **Don't drive the agent tool loop without ``--json``.** ``_agent._Spinner``
  writes ``\\r|/-\\ thinking… `` to stderr gated on ``sys.stderr.isatty()``,
  which is true under a pty, and it will pollute the buffer.
* **Pass ``--no-play`` to any audio command.** ``tts``/``music``/``sfx`` auto-play
  when stdout is a tty and a player binary exists.
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import os
import site
import subprocess
import sys
import warnings
from pathlib import Path

# Windows has no pty; the project is POSIX/macOS only (see the classifiers).
HAS_PEXPECT = (
    sys.platform != "win32" and importlib.util.find_spec("pexpect") is not None
)

SKIP_REASON = 'needs the test extra: pip install -e ".[test]"'

# Generous and uniform. Every drive assertion is an ordering assertion, never a
# timing one, so a slow CI runner just consumes more of the same headroom -- the
# only thing that can exhaust it is a genuine hang, which is what we want
# reported. No per-step tuning, and no sleep() anywhere in the harness.
DEFAULT_TIMEOUT = 20

# Reserved, never listening. Used when a test class runs without a fake server,
# so "no base URL" can never mean "the real api.venice.ai".
DEAD_BASE_URL = "http://127.0.0.1:1/api/v1"

# readline >= 8.1 wraps every prompt in bracketed-paste escapes
# (\x1b[?2004h ... \x1b[?2004l). Killing that at the source beats making every
# expected string escape-tolerant.
_INPUTRC = """\
set enable-bracketed-paste off
set bell-style none
set colored-stats off
set colored-completion-prefix off
"""


def _python_path() -> str:
    """PYTHONPATH giving the child the same imports this test process has.

    Two things to reproduce:

    * **venice itself** -- ``<repo>/src`` under ``make test``, or site-packages
      when pip-installed. Deriving it from the loaded package means the drive
      tests exercise whatever the runner is actually testing, either way.
    * **the site-packages dirs** -- because we redirect ``$HOME``, and on Linux
      the *user* site dir (``~/.local/lib/pythonX.Y/site-packages``) is derived
      from ``$HOME``. Without this the child silently loses anything installed
      with ``pip install --user``, and `venice chat` dies with "needs the openai
      package" instead of running the test.

      The user site dir is included only when this interpreter actually honours
      it (``site.ENABLE_USER_SITE``), which is **False** inside a venv -- the
      setup CONTRIBUTING.md tells contributors to use. Appending it regardless
      inverts the contract in the line above: the child would get *more* than
      the parent, out of a directory the parent deliberately opted out of, and
      resolve half a package from one interpreter's worldview and half from
      another's (#107).

    Deliberately **not** the whole of ``sys.path``: under ``make test`` that
    includes the repo root and ``tests/``, and PYTHONPATH precedes the stdlib on
    the child's path -- so a future fixture named after a stdlib module
    (``tests/queue.py``, ``tests/types.py``) would shadow it and break every
    drive test at once with an error pointing nowhere near the cause. It would
    also let the child satisfy imports from the repo root that a real install
    would not have, masking the packaging regressions the CI `package` job
    exists to catch.
    """
    spec = importlib.util.find_spec("venice")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("cannot locate the venice package")
    parts = [str(Path(spec.submodule_search_locations[0]).parent)]
    candidates = []
    if hasattr(site, "getsitepackages"):  # absent under some virtualenvs
        candidates.extend(site.getsitepackages())
    if site.ENABLE_USER_SITE and hasattr(site, "getusersitepackages"):
        candidates.append(site.getusersitepackages())
    for entry in candidates:
        if entry and os.path.isdir(entry) and entry not in parts:
            parts.append(entry)
    return os.pathsep.join(parts)


def build_env(home, *, base_url=None, api_key="test-fake-key") -> dict:
    """The child's environment, built from scratch rather than inherited.

    Inheriting would drag in ``http_proxy``/``ALL_PROXY`` -- both urllib and
    httpx honor them, and a CI proxy would swallow our 127.0.0.1 calls -- plus
    any ``VENICE_*`` a developer happens to have exported, which would silently
    change behavior between their machine and CI.
    """
    home = Path(home)
    (home / ".inputrc").write_text(_INPUTRC, encoding="utf-8")
    env = {
        "HOME": str(home),
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "PYTHONPATH": _python_path(),
        # Required: stdout is line-buffered on a tty, so without this the
        # streamed stdout deltas could surface *after* the stderr "usage:" line
        # and every ordering assertion would be a coin flip.
        "PYTHONUNBUFFERED": "1",
        # Portable UTF-8 (PEP 540) -- the CLI emits '…', '·' and friends, and CI
        # runners are not guaranteed a UTF-8 locale.
        "PYTHONUTF8": "1",
        "PYTHONIOENCODING": "utf-8",
        "PYTHONDONTWRITEBYTECODE": "1",
        # The user's effective site configuration is already represented in
        # PYTHONPATH above. Do not let the child's redirected HOME re-enable a
        # user site that the parent interpreter deliberately disabled.
        "PYTHONNOUSERSITE": "1",
        "TERM": "dumb",
        "COLUMNS": "200",
        "LINES": "24",
        # /edit must never drop into vi and hang the suite.
        "EDITOR": "true",
        "VISUAL": "true",
        "INPUTRC": str(home / ".inputrc"),
    }
    # Always pinned, never left to fall through to config.DEFAULT_BASE_URL --
    # otherwise a test class that runs without a fake (`server = False`) points
    # the child at https://api.venice.ai and hermeticity holds only because the
    # commands it drives happen not to make a call today. Port 1 is reserved and
    # never listening, so an unexpected request fails instantly and loudly
    # instead of quietly reaching the real API.
    env["VENICE_BASE_URL"] = base_url or DEAD_BASE_URL
    if api_key:
        env["VENICE_API_KEY"] = api_key
    # Deliberately NOT setting VENICE_SESSIONS_DIR / VENICE_MEMORY_DIR: config.py
    # derives them from $HOME at import, so the tmp home already sandboxes them
    # -- and leaving them off lets tests assert that a session envelope really
    # landed at <HOME>/.config/venice/sessions/.
    return env


class Driver:
    """A spawned CLI on a pty, with transcript capture for failure messages."""

    def __init__(self, child, sent_and_read: io.StringIO, read_only: io.StringIO):
        self.child = child
        self._all = sent_and_read
        self._read = read_only
        self._secrets = []

    # -- assertions --------------------------------------------------------

    def expect(self, needle: str, timeout=-1) -> None:
        """Wait for a literal substring, or fail with the whole transcript.

        ``timeout=-1`` is pexpect's "use the spawn's timeout" sentinel. It must
        not be ``None``: pexpect reads None as "block forever", which would make
        a missing prompt hang the suite instead of producing the transcript
        below -- defeating the harness on the exact failure it exists to report.
        """
        import pexpect

        try:
            self.child.expect_exact(needle, timeout=timeout)
        except (pexpect.TIMEOUT, pexpect.EOF) as exc:
            raise AssertionError(
                "drive: expected %r but never saw it (%s)\n"
                "--- transcript ---\n%s\n--- end transcript ---"
                % (needle, type(exc).__name__, self.transcript)
            ) from None

    def send(self, line: str) -> None:
        self.child.sendline(line)

    def send_partial(self, text: str) -> None:
        """Type without submitting -- no trailing newline.

        For testing what happens to a line the user never sent: Ctrl-C at the
        prompt is supposed to discard it (#92). ``send`` would submit it and turn
        the half-typed line into a real turn.
        """
        self.child.send(text)

    def send_secret(self, value: str) -> None:
        """Type a secret, and keep it out of any failure transcript.

        pexpect's ``logfile`` records sends as well as reads, so a plain
        ``send()`` at a hidden prompt would put the value straight into the
        AssertionError text of a failing test -- i.e. into CI logs and session
        transcripts, which CLAUDE.md names as the actual threat model, and which
        CONTRIBUTING.md's "never log, print, or embed the API key ... including
        in error messages" forbids. Use this for anything typed at a getpass
        prompt (`venice login`, `venice secret set`).
        """
        self._secrets.append(value)
        self.child.sendline(value)

    def send_eof(self) -> None:
        self.child.sendeof()

    def ctrl_c(self) -> None:
        self.child.sendintr()

    def wait(self) -> int:
        """Wait for exit and return the status.

        ``expect(EOF)`` first: ``close()`` defaults to ``force=True`` and would
        SIGKILL a still-running child, turning a real hang into a confusing
        ``signalstatus == 9``.
        """
        import pexpect

        try:
            self.child.expect(pexpect.EOF, timeout=self.child.timeout)
        except pexpect.TIMEOUT:
            raise AssertionError(
                "drive: the CLI never exited\n"
                "--- transcript ---\n%s\n--- end transcript ---" % self.transcript
            ) from None
        self.child.close()
        if self.child.signalstatus is not None:
            raise AssertionError(
                "drive: killed by signal %s\n"
                "--- transcript ---\n%s\n--- end transcript ---"
                % (self.child.signalstatus, self.transcript)
            )
        return self.child.exitstatus

    # -- captures ----------------------------------------------------------

    @property
    def transcript(self) -> str:
        """Everything sent and read, with registered secrets redacted.

        Redaction happens here rather than at the call sites so that every
        failure path -- expect(), wait(), and anything added later -- is covered
        by construction.
        """
        text = self._all.getvalue()
        for secret in self._secrets:
            text = text.replace(secret, "<redacted>")
        return text

    @property
    def screen(self) -> str:
        """Only what the terminal displayed.

        The correct basis for a "this was never echoed" assertion: `transcript`
        also contains what *we* typed, so it would trivially contain the secret.
        """
        return self._read.getvalue()


@contextlib.contextmanager
def cli(*argv, home, base_url=None, api_key="test-fake-key", cwd=None,
        timeout=DEFAULT_TIMEOUT):
    """Spawn ``python -m venice <argv>`` on a pty. Yields a :class:`Driver`."""
    import pexpect

    all_buf, read_buf = io.StringIO(), io.StringIO()
    # Python 3.12 warns on forkpty() from a multi-threaded process, and the fake
    # API server owns a thread. The hazard is a child that runs Python between
    # fork and exec while another thread holds an allocator lock; in practice
    # tests spawn before driving any traffic, so the only other thread is parked
    # in select() inside serve_forever, and pexpect execs promptly. That is a
    # strong likelihood, not a proof -- ThreadingHTTPServer can still be
    # unwinding a handler thread from an earlier test in the same class. The
    # backstop is `timeout-minutes: 15` in test.yml, so the theoretical deadlock
    # reports as a failed job rather than hanging a runner. Revisit by moving
    # the fake out of process (as _mcp_fake_server.py does) if it ever bites;
    # the cost would be losing in-process request assertions.
    # Scoped to this call so a genuine threading warning elsewhere still shows.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=".*multi-threaded.*forkpty.*",
            category=DeprecationWarning,
        )
        child = pexpect.spawn(
            sys.executable,
            ["-m", "venice", *argv],
            env=build_env(home, base_url=base_url, api_key=api_key),
            cwd=str(cwd) if cwd else None,
            timeout=timeout,
            encoding="utf-8",
            codec_errors="replace",
            # readline and argparse read the TIOCGWINSZ ioctl, not $COLUMNS, so
            # both have to be set or long lines wrap and break a match.
            dimensions=(24, 200),
        )
    # pexpect writes reads to both logfile and logfile_read, and sends to both
    # logfile and logfile_send -- so `all_buf` is the interleaved transcript and
    # `read_buf` is strictly what the terminal displayed.
    child.logfile = all_buf
    child.logfile_read = read_buf
    try:
        yield Driver(child, all_buf, read_buf)
    finally:
        # Never leave a child behind, even when an assertion fired mid-dialogue.
        with contextlib.suppress(Exception):
            child.close(force=True)


def run(*argv, home, base_url=None, api_key="test-fake-key", cwd=None,
        timeout=DEFAULT_TIMEOUT):
    """Run the CLI with **no** pty: pipes and ``stdin=/dev/null``.

    The ``isatty() is False`` branches (non-interactive charge refusal, `code`
    refusing to run unattended) are unreachable through a pty by definition.
    Needs no pexpect, so these cases keep running without the `[test]` extra.
    """
    return subprocess.run(
        [sys.executable, "-m", "venice", *argv],
        env=build_env(home, base_url=base_url, api_key=api_key),
        cwd=str(cwd) if cwd else None,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
