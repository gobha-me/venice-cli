"""Unit tests for the top-level dispatcher (`venice.cli.main`).

The dispatcher is four lines and used to have no exception handling at all, which
is how a Ctrl-C anywhere a command didn't catch it reached the interpreter as a raw
traceback (#92). These pin the net.
"""
import io
import sys
import unittest
from unittest import mock

from venice import cli


def _capture(fn, *args):
    out, err = io.StringIO(), io.StringIO()
    with mock.patch.object(sys, "stdout", out), mock.patch.object(sys, "stderr", err):
        rc = fn(*args)
        replacement = sys.stdout
    if replacement is not out:
        replacement.close()
    return rc, out.getvalue(), err.getvalue()


class _BrokenOnFlush(io.StringIO):
    def flush(self):
        raise BrokenPipeError


class TestDispatcher(unittest.TestCase):

    def test_keyboard_interrupt_from_a_handler_is_130_not_a_traceback(self):
        # #92: the backstop under the eight per-command Ctrl-C handlers. Any command
        # they miss used to propagate out of main(). `models` is just a stand-in for
        # "a handler that lets KeyboardInterrupt escape".
        with mock.patch("venice.commands.models._run", side_effect=KeyboardInterrupt):
            rc, out, err = _capture(cli.main, ["models"])
        self.assertEqual(rc, 130)

    def test_the_net_stays_quiet(self):
        # Deliberately silent apart from a newline: the command that owns the
        # interrupt has already printed its own notice (e.g. "code: aborted") and
        # this must not print a second one over it.
        with mock.patch("venice.commands.models._run", side_effect=KeyboardInterrupt):
            rc, out, err = _capture(cli.main, ["models"])
        self.assertEqual(err, "\n")
        self.assertEqual(out, "")

    def test_a_handler_that_returns_normally_is_untouched(self):
        with mock.patch("venice.commands.models._run", return_value=7):
            rc, out, err = _capture(cli.main, ["models"])
        self.assertEqual(rc, 7)

    def test_no_subcommand_still_prints_help_and_exits_2(self):
        rc, out, err = _capture(cli.main, [])
        self.assertEqual(rc, 2)
        self.assertIn("usage:", err)

    def test_other_exceptions_still_propagate(self):
        # The net is Ctrl-C only. A real bug must not be swallowed into an exit code.
        with mock.patch("venice.commands.models._run", side_effect=RuntimeError("boom")):
            with self.assertRaises(RuntimeError):
                _capture(cli.main, ["models"])

    def test_broken_pipe_from_a_handler_is_quiet_exit_141(self):
        with mock.patch("venice.commands.models._run", side_effect=BrokenPipeError):
            rc, out, err = _capture(cli.main, ["models"])
        self.assertEqual(rc, 141)
        self.assertEqual(out, "")
        self.assertEqual(err, "")

    def test_broken_pipe_from_the_final_flush_is_quiet_exit_141(self):
        out, err = _BrokenOnFlush(), io.StringIO()
        with mock.patch.object(sys, "stdout", out), mock.patch.object(sys, "stderr", err):
            with mock.patch("venice.commands.models._run", return_value=0):
                rc = cli.main(["models"])
            replacement = sys.stdout
        if replacement is not out:
            replacement.close()
        self.assertEqual(rc, 141)
        self.assertEqual(err.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
