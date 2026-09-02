import os
import tempfile
import unittest
from pathlib import Path

from venice import remote_media


PNG = b"\x89PNG\r\n\x1a\n" + b"x" * 64
MP3 = b"ID3" + b"x" * 64
MP4 = b"\x00\x00\x00\x18ftypisom" + b"x" * 64


class RemoteMediaStoreTests(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.now = 1000.0
        self.store = remote_media.RemoteMediaStore(
            self.td.name,
            public_origin="https://mcp.example.test",
            ttl_seconds=60,
            max_objects=3,
            principal_max_bytes=1024,
            global_max_bytes=2048,
            max_pending_jobs=1,
            global_max_pending_jobs=2,
            clock=lambda: self.now,
        )
        self.alice = remote_media.principal_key(
            issuer="https://auth.example.test", subject="alice", client_id="client-a"
        )
        self.bob = remote_media.principal_key(
            issuer="https://auth.example.test", subject="bob", client_id="client-a"
        )

    def tearDown(self):
        self.td.cleanup()

    def test_store_refuses_broad_or_relative_roots(self):
        for root in ("relative", "/", tempfile.gettempdir(), str(Path.home())):
            with self.subTest(root=root), self.assertRaises(remote_media.MediaStoreError):
                remote_media.RemoteMediaStore(
                    root, public_origin="https://mcp.example.test"
                )

    def _stage(self, data=PNG):
        path = self.store.new_temp_path()
        path.write_bytes(data)
        return path

    def _put(self, owner=None, data=PNG):
        return self.store.put_input(owner or self.alice, self._stage(data))

    def test_principal_key_requires_and_partitions_all_identity_components(self):
        with self.assertRaises(remote_media.MediaStoreError):
            remote_media.principal_key(issuer="", subject="alice", client_id="c")
        changed_client = remote_media.principal_key(
            issuer="https://auth.example.test", subject="alice", client_id="client-b"
        )
        self.assertNotEqual(self.alice, changed_client)
        self.assertNotEqual(self.alice, self.bob)

    def test_put_get_uri_and_delete_are_owner_bound(self):
        record = self._put()
        self.assertEqual(record.mime_type, "image/png")
        self.assertEqual(record.size, len(PNG))
        self.assertTrue(record.path.is_file())
        uri = self.store.resource_uri(record.id)
        self.assertEqual(self.store.get_uri(self.alice, uri).id, record.id)
        for operation in (
            lambda: self.store.get(self.bob, record.id),
            lambda: self.store.get_uri(self.bob, uri),
            lambda: self.store.delete(self.bob, record.id),
        ):
            with self.assertRaisesRegex(remote_media.MediaNotFound, "not found"):
                operation()
        self.store.delete(self.alice, record.id)
        self.assertFalse(record.path.exists())
        with self.assertRaises(remote_media.MediaNotFound):
            self.store.get(self.alice, record.id)

    def test_rejects_malformed_ids_mime_mismatch_and_unknown_content(self):
        with self.assertRaises(remote_media.MediaNotFound):
            self.store.id_from_uri("https://mcp.example.test/media/../secret")
        with self.assertRaises(remote_media.MediaValidationError):
            self.store.put_input(self.alice, self._stage(b"not-media"))
        staged = self._stage(PNG)
        reservation = self.store.reserve(self.alice, len(PNG))[0]
        with self.assertRaisesRegex(remote_media.MediaValidationError, "does not match"):
            self.store.commit_file(
                self.alice, reservation, staged, declared_mime="audio/mpeg"
            )
        self.store.release(self.alice, reservation)

    def test_reservation_enforces_count_owner_and_global_byte_quotas(self):
        reservations = self.store.reserve(self.alice, 100, count=3)
        with self.assertRaisesRegex(remote_media.MediaQuotaError, "object limit"):
            self.store.reserve(self.alice, 1)
        for reservation in reservations:
            self.store.release(self.alice, reservation)
        with self.assertRaisesRegex(remote_media.MediaQuotaError, "per-principal"):
            self.store.reserve(self.alice, 1025)
        first = self.store.reserve(self.alice, 1024)[0]
        second = self.store.reserve(self.bob, 1024)[0]
        with self.assertRaisesRegex(remote_media.MediaQuotaError, "global"):
            other = remote_media.principal_key(
                issuer="https://auth.example.test", subject="carol", client_id="client-a"
            )
            self.store.reserve(other, 1)
        self.store.release(self.alice, first)
        self.store.release(self.bob, second)

    def test_commit_requires_reservation_and_enforces_reserved_size(self):
        staged = self._stage(PNG)
        reservation = self.store.reserve(self.alice, len(PNG) - 1)[0]
        with self.assertRaisesRegex(remote_media.MediaQuotaError, "reserved"):
            self.store.commit_file(self.alice, reservation, staged)
        self.store.release(self.alice, reservation)
        with self.assertRaises(remote_media.MediaQuotaError):
            self.store.commit_file(self.alice, "A" * 43, staged)

    def test_expiry_is_fixed_and_prunes_blob_and_metadata(self):
        record = self._put()
        self.now += 59
        self.assertEqual(self.store.get(self.alice, record.id).expires_at, 1060)
        self.now += 2
        with self.assertRaises(remote_media.MediaNotFound):
            self.store.get(self.alice, record.id)
        self.assertFalse(record.path.exists())

    def test_restart_reconciles_partial_orphan_and_missing_files(self):
        record = self._put()
        orphan = self.store.blobs / ("A" * 43)
        orphan.write_bytes(PNG)
        partial = self.store.tmp / ".upload.part"
        partial.write_bytes(PNG)
        record.path.unlink()
        restarted = remote_media.RemoteMediaStore(
            self.td.name, public_origin="https://mcp.example.test", clock=lambda: self.now
        )
        self.assertFalse(orphan.exists())
        self.assertFalse(partial.exists())
        with self.assertRaises(remote_media.MediaNotFound):
            restarted.get(self.alice, record.id)

    def test_jobs_are_owner_bound_bounded_and_survive_restart(self):
        reservation = self.store.reserve(self.alice, 512)[0]
        job = self.store.create_job(
            self.alice, backend_id="queue-1", kind="video", model="v", 
            reservation_id=reservation, cost=0.05,
        )
        self.assertEqual(job.status, "processing")
        with self.assertRaises(remote_media.MediaNotFound):
            self.store.get_job(self.bob, job.id)
        other_reservation = self.store.reserve(self.alice, 100)[0]
        with self.assertRaisesRegex(remote_media.MediaQuotaError, "pending job"):
            self.store.create_job(
                self.alice, backend_id="queue-2", kind="sfx", model="a",
                reservation_id=other_reservation, cost=0.01,
            )
        self.store.release(self.alice, other_reservation)
        restarted = remote_media.RemoteMediaStore(
            self.td.name,
            public_origin="https://mcp.example.test",
            ttl_seconds=60,
            max_objects=3,
            principal_max_bytes=1024,
            global_max_bytes=2048,
            max_pending_jobs=1,
            global_max_pending_jobs=2,
            clock=lambda: self.now,
        )
        self.assertEqual(restarted.get_job(self.alice, job.id).backend_id, "queue-1")

    def test_job_completion_is_idempotent_and_failure_releases_reservation(self):
        reservation = self.store.reserve(self.alice, 512)[0]
        job = self.store.create_job(
            self.alice, backend_id="queue-1", kind="sfx", model="a",
            reservation_id=reservation, cost=None,
        )
        resource = self.store.commit_file(
            self.alice, reservation, self._stage(MP3), expected_kind="audio"
        )
        ready = self.store.finish_job(self.alice, job.id, resource.id)
        self.assertEqual(ready.status, "ready")
        self.assertEqual(self.store.finish_job(self.alice, job.id, resource.id), ready)

        reservation2 = self.store.reserve(self.bob, 512)[0]
        failed = self.store.create_job(
            self.bob, backend_id="queue-2", kind="video", model="v",
            reservation_id=reservation2, cost=0.1,
        )
        self.assertEqual(self.store.fail_job(self.bob, failed.id, "upstream failed").status, "failed")
        # A released reservation can no longer be committed.
        with self.assertRaises(remote_media.MediaQuotaError):
            self.store.commit_file(self.bob, reservation2, self._stage(MP4))

    def test_job_output_commit_atomically_publishes_ready_resource(self):
        reservation = self.store.reserve(self.alice, len(MP3))[0]
        job = self.store.create_job(
            self.alice, backend_id="queue-atomic", kind="sfx", model="a",
            reservation_id=reservation, cost=0.01,
        )
        record = self.store.commit_file(
            self.alice, reservation, self._stage(MP3), expected_kind="audio",
            job_id=job.id,
        )
        ready = self.store.get_job(self.alice, job.id)
        self.assertEqual(ready.status, "ready")
        self.assertEqual(ready.resource_id, record.id)
        self.assertIsNone(ready.reservation_id)

    def test_private_download_metadata_is_owner_bound(self):
        self.store.remember_download_url(
            self.alice, "queue-1", "model", "https://download.example/private?sig=x"
        )
        self.assertIsNone(self.store.lookup_download_url(self.bob, "queue-1", "model"))
        self.assertEqual(
            self.store.lookup_download_url(self.alice, "queue-1", "model"),
            "https://download.example/private?sig=x",
        )
        self.store.forget_download_url(self.alice, "queue-1", "model")
        self.assertIsNone(self.store.lookup_download_url(self.alice, "queue-1", "model"))
        with self.assertRaises(remote_media.MediaStoreError):
            self.store.remember_download_url(
                self.alice, "queue-1", "model", "http://download.example/private"
            )

    def test_store_permissions_are_private(self):
        self.assertEqual(os.stat(self.store.root).st_mode & 0o777, 0o700)
        self.assertEqual(os.stat(self.store.db_path).st_mode & 0o777, 0o600)
        record = self._put()
        self.assertEqual(os.stat(record.path).st_mode & 0o777, 0o600)


if __name__ == "__main__":
    unittest.main()
