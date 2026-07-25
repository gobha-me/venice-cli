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
    * **everything else on ``sys.path``** -- because we redirect ``$HOME``, and
      on Linux the *user* site-packages dir (``~/.local/lib/pythonX.Y/...``) is
      derived from ``$HOME``. Without this the child silently loses anything
      installed with ``pip install --user``, and `venice chat` dies with
      "needs the openai package" instead of running the test.

    ``''`` entries (the cwd) are dropped: the child gets an explicit ``cwd`` and
    must not import out of the project dir a test happens to be driving from.
    """
    spec = importlib.util.find_spec("venice")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("cannot locate the venice package")
    parts = [str(Path(spec.submodule_search_locations[0]).parent)]
    for entry in sys.path:
        if entry and os.path.isdir(entry) and entry not in parts:
            parts.append(entry)
    return os.pathsep.join(parts)


def build_env(home, *, base_url=None, api_key="test-fake-key", extra=None) -> dict:
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
        "TERM": "dumb",
        "COLUMNS": "200",
        "LINES": "24",
        # /edit must never drop into vi and hang the suite.
        "EDITOR": "true",
        "VISUAL": "true",
        "INPUTRC": str(home / ".inputrc"),
    }
    if base_url:
        env["VENICE_BASE_URL"] = base_url
    if api_key:
        env["VENICE_API_KEY"] = api_key
    # Deliberately NOT setting VENICE_SESSIONS_DIR / VENICE_MEMORY_DIR: config.py
    # derives them from $HOME at import, so the tmp home already sandboxes them
    # -- and leaving them off lets tests assert that a session envelope really
    # landed at <HOME>/.config/venice/sessions/.
    env.update(extra or {})
    return env


class Driver:
    """A spawned CLI on a pty, with transcript capture for failure messages."""

    def __init__(self, child, sent_and_read: io.StringIO, read_only: io.StringIO):
        self.child = child
        self._all = sent_and_read
        self._read = read_only

    # -- assertions --------------------------------------------------------

    def expect(self, needle: str, timeout=None) -> None:
        """Wait for a literal substring, or fail with the whole transcript."""
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
        """Everything sent and read -- for failure messages."""
        return self._all.getvalue()

    @property
    def screen(self) -> str:
        """Only what the terminal displayed.

        The correct basis for a "this was never echoed" assertion: `transcript`
        also contains what *we* typed, so it would trivially contain the secret.
        """
        return self._read.getvalue()


@contextlib.contextmanager
def cli(*argv, home, base_url=None, api_key="test-fake-key", cwd=None,
        timeout=DEFAULT_TIMEOUT, env_extra=None):
    """Spawn ``python -m venice <argv>`` on a pty. Yields a :class:`Driver`."""
    import pexpect

    all_buf, read_buf = io.StringIO(), io.StringIO()
    # Python 3.12 warns on forkpty() from a multi-threaded process, and the fake
    # API server owns a thread. The warning is about a child that runs Python
    # code between fork and exec while another thread holds a lock; here the only
    # other thread is parked in select() inside serve_forever (tests spawn before
    # driving any traffic) and pexpect execs immediately. Scoped to this call so
    # a genuine threading warning elsewhere still surfaces.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=".*multi-threaded.*forkpty.*",
            category=DeprecationWarning,
        )
        child = pexpect.spawn(
            sys.executable,
            ["-m", "venice", *argv],
            env=build_env(home, base_url=base_url, api_key=api_key, extra=env_extra),
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


def run(*argv, home, base_url=None, api_key="test-fake-key", cwd=None, timeout=60):
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
