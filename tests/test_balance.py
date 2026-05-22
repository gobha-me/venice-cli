"""Unit tests for `venice balance` (mocks urlopen)."""
import argparse
import io
import json
import os
import sys
import unittest
from unittest import mock

from tests.test_client import FakeResp


_FAKE_DATA = {
    "data": {
        "accessPermitted": True,
        "apiTier": {"id": "paid", "isCharged": True},
        "balances": {"USD": 12.3456, "DIEM": 1.234},
        "nextEpochBegins": "2026-05-23T00:00:00.000Z",
        "keyExpiration": None,
        "rateLimits": [],
    }
}


def _args(**ov):
    base = dict(json=False, verbose=False, min=None)
    base.update(ov)
    return argparse.Namespace(**base)


def _patched_env():
    return mock.patch.dict(os.environ, {"VENICE_API_KEY": "fake"})


class TestBalance(unittest.TestCase):

    def test_default_prints_combined_total_to_stdout(self):
        from venice.commands import balance

        buf = io.StringIO()
        with _patched_env(), \
             mock.patch(
                 "venice.client.urllib.request.urlopen",
                 lambda *a, **kw: FakeResp(
                     200, json.dumps(_FAKE_DATA).encode(), "application/json"
                 ),
             ), \
             mock.patch.object(sys, "stdout", buf):
            rc = balance._run(_args())
        self.assertEqual(rc, 0)
        # USD 12.3456 + DIEM 1.234 = 13.5796 -> "$13.58 USD"
        self.assertIn("$13.58 USD", buf.getvalue())

    def test_verbose_shows_breakdown(self):
        from venice.commands import balance

        buf = io.StringIO()
        with _patched_env(), \
             mock.patch(
                 "venice.client.urllib.request.urlopen",
                 lambda *a, **kw: FakeResp(
                     200, json.dumps(_FAKE_DATA).encode(), "application/json"
                 ),
             ), \
             mock.patch.object(sys, "stdout", buf):
            rc = balance._run(_args(verbose=True))
        self.assertEqual(rc, 0)
        out = buf.getvalue()
        self.assertIn("Spendable:", out)
        self.assertIn("USD", out)
        self.assertIn("DIEM", out)
        self.assertIn("$13.58", out)
        self.assertIn("12.35 USD", out)
        self.assertIn("1.23 DIEM credit", out)

    def test_json_includes_total_tier_and_epoch(self):
        from venice.commands import balance

        buf = io.StringIO()
        with _patched_env(), \
             mock.patch(
                 "venice.client.urllib.request.urlopen",
                 lambda *a, **kw: FakeResp(
                     200, json.dumps(_FAKE_DATA).encode(), "application/json"
                 ),
             ), \
             mock.patch.object(sys, "stdout", buf):
            rc = balance._run(_args(json=True))
        self.assertEqual(rc, 0)
        out = json.loads(buf.getvalue())
        self.assertEqual(out["USD"], 12.3456)
        self.assertEqual(out["DIEM"], 1.234)
        self.assertAlmostEqual(out["total_usd_equiv"], 12.3456 + 1.234, places=6)
        self.assertEqual(out["tier"], "paid")
        self.assertEqual(out["next_epoch"], "2026-05-23T00:00:00.000Z")

    def test_min_threshold_below_returns_exit_1(self):
        from venice.commands import balance

        buf = io.StringIO()
        with _patched_env(), \
             mock.patch(
                 "venice.client.urllib.request.urlopen",
                 lambda *a, **kw: FakeResp(
                     200, json.dumps(_FAKE_DATA).encode(), "application/json"
                 ),
             ), \
             mock.patch.object(sys, "stdout", buf):
            rc = balance._run(_args(min=100.0))
        self.assertEqual(rc, 1)

    def test_min_threshold_above_returns_exit_0(self):
        from venice.commands import balance

        buf = io.StringIO()
        with _patched_env(), \
             mock.patch(
                 "venice.client.urllib.request.urlopen",
                 lambda *a, **kw: FakeResp(
                     200, json.dumps(_FAKE_DATA).encode(), "application/json"
                 ),
             ), \
             mock.patch.object(sys, "stdout", buf):
            rc = balance._run(_args(min=5.0))
        self.assertEqual(rc, 0)

    def test_missing_api_key_returns_exit_2(self):
        from venice.commands import balance
        import importlib
        import tempfile

        tmp = tempfile.TemporaryDirectory()
        with mock.patch.dict(os.environ, {"HOME": tmp.name}, clear=False):
            os.environ.pop("VENICE_API_KEY", None)
            import venice.config as _cfg
            import venice.auth as _auth
            importlib.reload(_cfg)
            importlib.reload(_auth)
            rc = balance._run(_args())
        tmp.cleanup()
        self.assertEqual(rc, 2)


if __name__ == "__main__":
    unittest.main()
