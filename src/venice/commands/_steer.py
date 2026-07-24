"""Attached-terminal steering for a running agent (#79).

#78 gave a running ``venice code`` agent a *detached* steering channel: the file
mailbox (:mod:`_mailbox`) + ``venice sessions send``, drained by :func:`_agent.run_loop`
at the top-of-turn checkpoint. This module adds the *attached* half -- steering the run
you're watching in a terminal with **Ctrl+C**, without dropping to another shell.

The whole feature rides on the same seam #78 added, so ``run_loop`` itself is untouched:

* :func:`pause_and_steer` is a context manager the attached driver wraps around its
  ``run_loop`` call. On an interactive tty it installs a scoped SIGINT handler and hands
  back a ``steer_drain`` callable; anywhere else it hands back the plain #78 mailbox
  drain (or ``None``) and installs nothing -- so non-tty / ``--json`` / detached runs
  behave byte-for-byte as before.
* **First Ctrl+C** merely *arms* a steer (the handler sets a flag and returns). Per
  PEP 475 the interrupted blocking syscall -- the model call's socket read, or a ``run``
  tool's subprocess -- is retried and finishes, so the current operation completes
  rather than being torn out. At the next checkpoint the drain prompts on the tty and
  feeds a non-empty line back through the exact same ``[steering message received
  mid-run]`` path as a mailbox steer.
* **Second Ctrl+C** (the operator mashing it to kill), or Ctrl+C *at* the prompt,
  raises ``KeyboardInterrupt`` -- today's abort. The attached driver catches it (one-shot
  exits 130 + saves the partial transcript; the REPL rolls the turn back).

Trust model (CLAUDE.md): a steer is additive operator input at the same trust level as
the original task -- nothing here reads or writes the API key, and the handler only
touches an in-process flag + stderr.
"""
from __future__ import annotations

import signal
import sys
from contextlib import contextmanager
from typing import Callable, List, Optional

from . import _mailbox

# Written by the handler on the first Ctrl+C so the operator sees that the interrupt
# landed even though the current (possibly slow) operation is still finishing.
_HINT = (
    "\n[steer] Ctrl+C caught -- will prompt at the next checkpoint "
    "(Ctrl+C again to abort).\n"
)
_PROMPT = "[paused] message to the agent (empty = resume, Ctrl+C = abort): "

SteerDrain = Optional[Callable[[], List[str]]]


class _Pending:
    """One-slot armed flag shared between the SIGINT handler and the drain callable."""

    __slots__ = ("requested",)

    def __init__(self) -> None:
        self.requested = False


def _make_handler(state: "_Pending"):
    """A SIGINT handler that arms a steer on the first hit and aborts on the second.

    First Ctrl+C: set ``requested`` and return (the in-flight syscall resumes, PEP 475).
    Second Ctrl+C before the checkpoint consumed the first: re-raise so an operator
    mashing Ctrl+C still gets today's immediate kill.
    """

    def _handler(signum, frame):
        if state.requested:
            raise KeyboardInterrupt
        state.requested = True
        try:
            sys.stderr.write(_HINT)
            sys.stderr.flush()
        except Exception:
            pass

    return _handler


def _plain_drain(session_id) -> SteerDrain:
    """The #78 behavior: drain only the mailbox (or nothing without a session)."""
    if not session_id:
        return None
    return lambda sid=session_id: _mailbox.drain(sid)


@contextmanager
def default_sigint():
    """Temporarily restore the default (KeyboardInterrupt-raising) SIGINT handler.

    For an interactive prompt that must let Ctrl+C abort *even while* the attached-steer
    handler (:func:`pause_and_steer`) is installed -- the steer prompt itself, and
    ``run_loop``'s paid-tool confirm (:func:`_agent._prompt_yes`). The prior handler is
    restored on exit; a best-effort no-op off the main thread or when signals are
    unavailable.
    """
    try:
        prev = signal.getsignal(signal.SIGINT)
    except (ValueError, OSError):
        yield
        return
    swapped = False
    try:
        try:
            signal.signal(signal.SIGINT, signal.default_int_handler)
            swapped = True
        except (ValueError, OSError, TypeError, RuntimeError):
            pass
        yield
    finally:
        if swapped:
            try:
                signal.signal(
                    signal.SIGINT, prev if prev is not None else signal.SIG_DFL
                )
            except (ValueError, OSError, TypeError, RuntimeError):
                pass


def _read_steer_line(prompt: Optional[Callable[[str], str]] = None) -> str:
    """Read one steering line from the tty; ``""`` means "resume, no steer".

    Ctrl+C *here* must abort, not arm another steer, so :func:`default_sigint` swaps the
    steering handler for the default (KeyboardInterrupt-raising) one for the duration of
    the read. ^D (EOF) resumes without steering. ``prompt`` defaults to the builtin
    ``input`` resolved at call time (so a test's patched ``input`` is honored).
    """
    reader = prompt if prompt is not None else input
    with default_sigint():
        try:
            return reader(_PROMPT)
        except EOFError:
            return ""


def _make_drain(session_id, state: "_Pending",
                prompt: Optional[Callable[[str], str]] = None) -> Callable[[], List[str]]:
    """Compose the #78 mailbox drain with a tty prompt gated on the armed flag.

    Returns the drained mailbox messages plus, when a Ctrl+C armed a steer, whatever
    the operator types at the prompt (inline -- no ``_mailbox.deposit`` round-trip, since
    the returned list is appended by ``run_loop`` on its single steer path anyway).
    """

    def _drain() -> List[str]:
        out = _mailbox.drain(session_id) if session_id else []
        if state.requested:
            state.requested = False
            line = _read_steer_line(prompt)
            if line and line.strip():
                out.append(line)
        return out

    return _drain


@contextmanager
def pause_and_steer(session_id, *, enabled: bool):
    """Yield a ``steer_drain`` for :func:`_agent.run_loop`, tty-steering when ``enabled``.

    When ``enabled`` (an attached, interactive tty), install a scoped SIGINT handler and
    yield a drain that pauses+prompts on the first Ctrl+C. Otherwise -- non-tty,
    ``--json``, or a thread where ``signal.signal`` is unavailable -- yield the plain #78
    mailbox drain (or ``None``) and install no handler, so behavior is unchanged. The
    prior SIGINT disposition is always restored on exit, including on an abort unwinding
    through the ``with`` body.
    """
    if not enabled:
        yield _plain_drain(session_id)
        return
    try:
        prev = signal.getsignal(signal.SIGINT)
    except (ValueError, OSError):
        yield _plain_drain(session_id)
        return
    state = _Pending()
    try:
        signal.signal(signal.SIGINT, _make_handler(state))
    except (ValueError, OSError, TypeError, RuntimeError):
        # Not the main thread (e.g. a subagent loop) -- can't steer here; degrade to #78.
        yield _plain_drain(session_id)
        return
    try:
        yield _make_drain(session_id, state)
    finally:
        try:
            signal.signal(signal.SIGINT, prev if prev is not None else signal.SIG_DFL)
        except (ValueError, OSError, TypeError, RuntimeError):
            pass
