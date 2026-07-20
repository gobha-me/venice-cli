"""Unit tests for the shared image-input helpers in `_shared`.

`encode_base64` and `check_image_file` were lifted out of `upscale`/`bg-remove`/
`image-edit` (GitHub #34), which each carried a byte-identical copy. Those
commands exercise them only end-to-end; this covers the pure logic directly:
raw-base64 round-trip and the exists / non-empty / size gate with its
`label`-prefixed stderr. No network, no real key.
"""
import base64
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from venice.commands import _shared


class TestEncodeBase64(unittest.TestCase):

    def test_round_trips_raw_base64_no_prefix(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "in.png"
            p.write_bytes(b"\x89PNG\r\n\x1a\nRAWBYTES")
            out = _shared.encode_base64(p)
            # raw base64: no `data:` prefix (that is encode_data_url's job)
            self.assertFalse(out.startswith("data:"))
            self.assertEqual(base64.b64decode(out), b"\x89PNG\r\n\x1a\nRAWBYTES")

    def test_empty_file_encodes_to_empty_string(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "empty.png"
            p.write_bytes(b"")
            self.assertEqual(_shared.encode_base64(p), "")


class TestCheckImageFile(unittest.TestCase):

    def _check(self, path, *, label="upscale", max_bytes=_shared.MAX_IMAGE_BYTES):
        err = io.StringIO()
        with mock.patch.object(sys, "stderr", err):
            rc = _shared.check_image_file(path, label=label, max_bytes=max_bytes)
        return rc, err.getvalue()

    def test_good_file_returns_none_no_output(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "ok.png"
            p.write_bytes(b"data")
            rc, err = self._check(p)
            self.assertIsNone(rc)
            self.assertEqual(err, "")

    def test_missing_file_exits_2_with_label(self):
        rc, err = self._check(Path("nope.png"), label="bg-remove")
        self.assertEqual(rc, 2)
        self.assertTrue(err.startswith("bg-remove: "))
        self.assertIn("input file not found", err)

    def test_empty_file_exits_2_with_label(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "empty.png"
            p.write_bytes(b"")
            rc, err = self._check(p, label="image-edit")
            self.assertEqual(rc, 2)
            self.assertTrue(err.startswith("image-edit: "))
            self.assertIn("is empty", err)

    def test_oversized_file_exits_2_and_reports_mb(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "big.png"
            p.write_bytes(b"0123456789")  # 10 bytes
            rc, err = self._check(p, label="upscale", max_bytes=5)
            self.assertEqual(rc, 2)
            self.assertTrue(err.startswith("upscale: "))
            self.assertIn("10 bytes", err)

    def test_default_cap_message_says_25_mb(self):
        # The default MAX_IMAGE_BYTES must render as "< 25 MB" (byte-identical to
        # the pre-refactor literal the three commands printed).
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "big.png"
            p.write_bytes(b"x" * (_shared.MAX_IMAGE_BYTES + 1))
            rc, err = self._check(p, label="upscale")
            self.assertEqual(rc, 2)
            self.assertIn("must be < 25 MB", err)


if __name__ == "__main__":
    unittest.main()
