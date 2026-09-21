"""Lossless context archive tests for issue #74 (no network or credentials)."""
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from venice.commands import _context_archive as A


class TestContextArchive(unittest.TestCase):
    @staticmethod
    def _read_all(archive, entry_id):
        offset = 0
        parts = []
        while True:
            page = archive.read(entry_id, offset)
            parts.append(page["content"])
            if page["complete"]:
                return "".join(parts)
            offset = page["next_offset"]

    def test_stage_commit_list_and_exact_paged_read(self):
        message = {"role": "tool", "name": "read_file", "content": "🐍" * 10000,
                   "tool_call_id": "call-1"}
        archive = A.ContextArchive()
        staged = archive.stage([message])
        self.assertEqual(archive.entries, [])
        archive.commit(staged)

        listing = archive.list_page()
        self.assertEqual(listing["total"], 1)
        self.assertEqual(listing["entries"][0]["name"], "read_file")
        first = archive.read("ctx-000001")
        self.assertFalse(first["complete"])
        self.assertLessEqual(len(first["content"].encode("utf-8")), A.MAX_READ_BYTES)
        second = archive.read("ctx-000001", first["next_offset"])
        exact = first["content"] + second["content"]
        self.assertEqual(json.loads(exact), message)
        self.assertEqual(first["sha256"], listing["entries"][0]["sha256"])
        self.assertNotIn("message", archive.to_envelope()[0])

    def test_limits_are_fail_closed_and_leave_archive_unchanged(self):
        archive = A.ContextArchive()
        archive.commit(archive.stage([{"role": "user", "content": "kept"}]))
        before = archive.to_envelope()
        with mock.patch.object(A, "MAX_ARCHIVE_ENTRIES", 1):
            with self.assertRaisesRegex(A.ArchiveError, "archive full"):
                archive.stage([{"role": "assistant", "content": "overflow"}])
        self.assertEqual(archive.to_envelope(), before)

        with mock.patch.object(A, "MAX_ARCHIVE_BYTES", archive.bytes_used + 1):
            with self.assertRaisesRegex(A.ArchiveError, "archive full"):
                archive.stage([{"role": "user", "content": "too large"}])
        self.assertEqual(archive.to_envelope(), before)

    def test_envelope_validation_rejects_tampering(self):
        archive = A.ContextArchive()
        archive.commit(archive.stage([{"role": "assistant", "content": "answer"}]))
        raw = archive.to_envelope()
        raw[0]["sha256"] = "not-a-digest"
        with self.assertRaisesRegex(A.ArchiveError, "invalid sha256"):
            A.ContextArchive.from_envelope(raw)

    def test_large_multimodal_message_uses_private_sidecar_not_envelope(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive = A.ContextArchive()
            store = Path(tmp) / "archive"
            archive.bind(store)
            payload = "A" * (15 * 1024 * 1024)
            message = {
                "role": "user",
                "content": [
                    {"type": "text", "text": "inspect"},
                    {"type": "image_url", "image_url": {
                        "url": "data:image/png;base64," + payload,
                    }},
                ],
            }
            archive.commit(archive.stage([message]))
            envelope = archive.to_envelope()
            self.assertNotIn("message", envelope[0])
            self.assertLess(len(json.dumps(envelope)), 2048)
            blob = store / envelope[0]["blob"]
            self.assertTrue(blob.is_file())
            self.assertEqual(os.stat(blob).st_mode & 0o777, 0o600)
            self.assertEqual(json.loads(self._read_all(archive, "ctx-000001")), message)

    def test_v2_inline_entry_migrates_when_bound(self):
        message = {"role": "user", "content": "legacy exact evidence"}
        entry = A.ContextArchive._entry("ctx-000001", message)
        legacy = {key: entry[key] for key in (
            "id", "role", "chars", "sha256", "head", "tail"
        )}
        legacy["message"] = message
        with tempfile.TemporaryDirectory() as tmp:
            archive = A.ContextArchive.from_envelope([legacy])
            archive.bind(Path(tmp) / "archive")
            self.assertNotIn("message", archive.to_envelope()[0])
            self.assertEqual(
                json.loads(self._read_all(archive, "ctx-000001")), message,
            )

    def test_missing_or_corrupt_blob_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Path(tmp) / "archive"
            archive = A.ContextArchive()
            archive.bind(store)
            archive.commit(archive.stage([{"role": "user", "content": "proof"}]))
            envelope = archive.to_envelope()
            blob = store / envelope[0]["blob"]
            blob.write_bytes(b"x" * envelope[0]["bytes"])
            restored = A.ContextArchive.from_envelope(envelope, source_dir=store)
            restored.bind(store)
            with self.assertRaisesRegex(A.ArchiveError, "failed verification"):
                restored.read("ctx-000001")

    def test_ephemeral_storage_is_removed_on_close(self):
        archive = A.ContextArchive()
        archive.bind_temporary()
        root = archive.storage_dir
        archive.commit(archive.stage([{"role": "user", "content": "temporary"}]))
        self.assertTrue(root.is_dir())
        archive.close()
        self.assertFalse(root.exists())

    def test_list_and_read_validate_bounds(self):
        archive = A.ContextArchive()
        with self.assertRaises(A.ArchiveError):
            archive.list_page(-1)
        with self.assertRaises(A.ArchiveError):
            archive.list_page(limit=51)
        with self.assertRaises(A.ArchiveError):
            archive.read("missing")

    def test_live_index_is_bounded_but_older_metadata_remains_listable(self):
        archive = A.ContextArchive()
        archive.commit(archive.stage([
            {"role": "user", "content": str(i)}
            for i in range(A.MAX_LIST_ENTRIES + 5)
        ]))
        text = archive.live_index_message()["content"]
        self.assertNotIn("ctx-000001 user", text)
        self.assertIn("23 older", text)
        first = archive.list_page()
        self.assertEqual(len(first["entries"]), A.MAX_LIST_ENTRIES)
        self.assertEqual(first["next_cursor"], A.MAX_LIST_ENTRIES)
        second = archive.list_page(first["next_cursor"])
        self.assertEqual(len(second["entries"]), 5)
        self.assertIsNone(second["next_cursor"])

    def test_tool_is_read_only_and_uses_current_archive(self):
        archive = A.ContextArchive()
        tool = A.archive_tool(archive)
        self.assertFalse(tool.paid)
        self.assertEqual(tool.tags, ("read",))
        archive.commit(archive.stage([{"role": "user", "content": "later"}]))
        result = tool.invoke({"action": "list"}, confirm=False)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["total"], 1)


if __name__ == "__main__":
    unittest.main()
