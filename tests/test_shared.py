"""Unit tests for the shared image-input helpers in `_shared`.

`encode_base64` and `check_image_file` were lifted out of `upscale`/`bg-remove`/
`image-edit` (GitHub #34), which each carried a byte-identical copy. Those
commands exercise them only end-to-end; this covers the pure logic directly:
raw-base64 round-trip and the exists / non-empty / size gate with its
`label`-prefixed stderr. No network, no real key.
"""
import base64
import io
import os
import shutil
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


class TestMediaPathAuthority(unittest.TestCase):

    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.outside = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.outside, ignore_errors=True)
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.authority = _shared.MediaPathAuthority(self.root)

    def _write(self, directory, name, data):
        path = Path(directory) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path

    def test_recognizes_supported_signatures_and_ignores_false_extension(self):
        cases = (
            ("image", b"\x89PNG\r\n\x1a\nbody", "image/png"),
            ("image", b"\xff\xd8\xff\xe0body", "image/jpeg"),
            ("image", b"GIF89abody", "image/gif"),
            ("image", b"RIFFxxxxWEBPbody", "image/webp"),
            ("video", b"\x00\x00\x00\x18ftypisombody", "video/mp4"),
            ("video", b"\x1aE\xdf\xa3body", "video/webm"),
            ("audio", b"ID3body", "audio/mpeg"),
            ("audio", b"\xff\xfbbody", "audio/mpeg"),
            ("audio", b"\xff\xf1body", "audio/aac"),
            ("audio", b"RIFFxxxxWAVEbody", "audio/wav"),
            ("audio", b"fLaCbody", "audio/flac"),
            ("audio", b"OggSbody", "audio/ogg"),
        )
        for index, (kind, data, expected) in enumerate(cases):
            with self.subTest(kind=kind, expected=expected):
                path = self._write(self.root, f"media-{index}.txt", data)
                resolved, mime = self.authority.resolve(
                    path, kind=kind, max_bytes=1024
                )
                self.assertEqual(resolved, path.resolve())
                self.assertEqual(mime, expected)

    def test_rejects_outside_root_and_symlink_escape(self):
        target = self._write(
            self.outside, "outside.png", b"\x89PNG\r\n\x1a\nbody"
        )
        link = Path(self.root) / "inside.png"
        os.symlink(target, link)
        for candidate in (target, link):
            with self.subTest(candidate=candidate):
                with self.assertRaisesRegex(_shared.MediaPathError, "escapes"):
                    self.authority.resolve(
                        candidate, kind="image", max_bytes=1024
                    )

    def test_rejects_secret_and_protected_paths(self):
        for name in ("credentials", ".env", ".git/object.png", ".venice/key.png"):
            path = self._write(
                self.root, name, b"\x89PNG\r\n\x1a\nbody"
            )
            with self.subTest(name=name):
                with self.assertRaisesRegex(_shared.MediaPathError, "protected"):
                    self.authority.resolve(path, kind="image", max_bytes=1024)

    def test_rejects_missing_directory_empty_oversized_and_non_media(self):
        directory = Path(self.root) / "folder"
        directory.mkdir()
        empty = self._write(self.root, "empty.png", b"")
        large = self._write(self.root, "large.png", b"\x89PNG\r\n\x1a\nxxx")
        text = self._write(self.root, "fake.png", b"plain text")
        cases = (
            (Path(self.root) / "missing.png", "does not exist", 1024),
            (directory, "regular file", 1024),
            (empty, "empty", 1024),
            (large, "limit", 8),
            (text, "recognized image", 1024),
        )
        for candidate, message, limit in cases:
            with self.subTest(candidate=candidate):
                with self.assertRaisesRegex(_shared.MediaPathError, message):
                    self.authority.resolve(
                        candidate, kind="image", max_bytes=limit
                    )

    def test_dynamic_active_and_attached_roots(self):
        other = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, other, ignore_errors=True)
        media = self._write(other, "frame", b"\x89PNG\r\n\x1a\nbody")
        state = {"base": self.root, "roots": [self.root]}
        authority = _shared.MediaPathAuthority(
            lambda: state["base"], lambda: state["roots"]
        )
        with self.assertRaises(_shared.MediaPathError):
            authority.resolve(media, kind="image", max_bytes=1024)
        state["roots"].append(other)
        state["base"] = other
        resolved, mime = authority.resolve("frame", kind="image", max_bytes=1024)
        self.assertEqual(resolved, media.resolve())
        self.assertEqual(mime, "image/png")


if __name__ == "__main__":
    unittest.main()
