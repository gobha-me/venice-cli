"""Private background-video registry tests (#140)."""
import json
import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from venice import config
from venice.commands import _video_jobs


class TestVideoJobRegistry(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "jobs.json"
        env = mock.patch.dict(
            os.environ, {config.ENV_VIDEO_JOBS_FILE: str(self.path)}
        )
        env.start()
        self.addCleanup(env.stop)

    def test_round_trip_is_mode_0600_and_bound_to_queue_and_model(self):
        url = "https://cdn.example/video?signature=secret"
        _video_jobs.remember("queue-1", "model-a", url)
        self.assertEqual(stat.S_IMODE(self.path.stat().st_mode), 0o600)
        self.assertEqual(_video_jobs.lookup("queue-1", "model-a"), url)
        self.assertIsNone(_video_jobs.lookup("queue-1", "model-b"))
        self.assertIsNone(_video_jobs.lookup("queue-2", "model-a"))

    def test_forget_removes_only_the_exact_binding(self):
        _video_jobs.remember("queue-1", "model-a", "https://a.example/v?x=1")
        _video_jobs.remember("queue-2", "model-a", "https://b.example/v?x=2")
        _video_jobs.forget("queue-1", "model-a")
        self.assertIsNone(_video_jobs.lookup("queue-1", "model-a"))
        self.assertIsNotNone(_video_jobs.lookup("queue-2", "model-a"))

    def test_expired_entries_are_pruned_lazily(self):
        with mock.patch("venice.commands._video_jobs.time.time", return_value=100.0):
            _video_jobs.remember("old", "m", "https://old.example/v?secret=x")
        with mock.patch(
            "venice.commands._video_jobs.time.time",
            return_value=100.0 + _video_jobs.MAX_AGE_SECONDS + 1,
        ):
            self.assertIsNone(_video_jobs.lookup("old", "m"))
        self.assertEqual(json.loads(self.path.read_text())["jobs"], {})

    def test_registry_is_bounded_to_newest_entries(self):
        with mock.patch.object(_video_jobs, "MAX_JOBS", 2):
            for i in range(3):
                with mock.patch(
                    "venice.commands._video_jobs.time.time", return_value=float(i + 1)
                ):
                    _video_jobs.remember(
                        f"q{i}", "m", f"https://cdn.example/{i}?signature=secret"
                    )
        with mock.patch("venice.commands._video_jobs.time.time", return_value=3.0):
            self.assertIsNone(_video_jobs.lookup("q0", "m"))
            self.assertIsNotNone(_video_jobs.lookup("q1", "m"))
            self.assertIsNotNone(_video_jobs.lookup("q2", "m"))

    def test_malformed_store_fails_closed_without_overwrite(self):
        self.path.write_text("not json", encoding="utf-8")
        before = self.path.read_bytes()
        with self.assertRaises(_video_jobs.VideoJobStoreError):
            _video_jobs.remember("q", "m", "https://cdn.example/v?secret=x")
        self.assertEqual(self.path.read_bytes(), before)

    def test_non_https_url_is_rejected_without_storing_or_disclosing_query(self):
        url = "http://127.0.0.1/private?signature=secret"
        with self.assertRaises(_video_jobs.VideoJobStoreError) as cm:
            _video_jobs.remember("q", "m", url)
        self.assertFalse(self.path.exists())
        self.assertNotIn("signature", str(cm.exception))
        self.assertNotIn("secret", str(cm.exception))

    def test_custom_parent_permissions_are_not_changed(self):
        parent = self.path.parent
        parent.chmod(0o755)
        _video_jobs.remember("q", "m", "https://cdn.example/v?secret=x")
        self.assertEqual(stat.S_IMODE(parent.stat().st_mode), 0o755)


if __name__ == "__main__":
    unittest.main()
