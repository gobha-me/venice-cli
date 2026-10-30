"""Regressions for deprecated video cleanup fields and diagnostic credentials."""
import io
import json
import os
import unittest
import urllib.error
from types import SimpleNamespace
from unittest import mock

from venice import _redact
from venice.client import VeniceAPIError, VeniceClient
from venice.commands import _mcp, _openai, video


class CleanupTests(unittest.TestCase):
    def test_video_completion_uses_queue_id_and_retrieve_keeps_model(self):
        self.assertEqual(video.complete_body("job"), {"queue_id": "job"})
        self.assertEqual(video.job_body("model", "job"), {"model": "model", "queue_id": "job"})

    def test_cleanup_delegation_does_not_emit_deprecated_model(self):
        client = mock.Mock()
        jobs = mock.Mock()
        _mcp.complete_job(client, queue_id="job", type="video", model="model", job_store=jobs)
        client.post_json.assert_called_once_with("/video/complete", {"queue_id": "job"})
        jobs.forget.assert_called_once_with("job", "model")

    def test_audio_completion_still_requires_model(self):
        client = mock.Mock()
        _mcp.complete_job(client, queue_id="job", type="sfx", model="model")
        client.post_json.assert_called_once_with("/audio/complete", {"model": "model", "queue_id": "job"})


class RedactionTests(unittest.TestCase):
    def test_explicit_and_environment_secrets_are_removed_in_encoded_forms(self):
        secret = 'audit-only/with+"quote'
        with mock.patch.dict(os.environ, {"VENICE_API_KEY": secret}):
            for value in (secret, json.dumps(secret)[1:-1], "audit-only%2Fwith%2B%22quote"):
                self.assertEqual(_redact.redact_text(value), "[REDACTED]")
        self.assertEqual(_redact.redact_text("hello credential", "credential"), "hello [REDACTED]")

    def test_http_error_redacts_explicit_key_before_excerpt_truncation(self):
        secret = "synthetic-file-loaded-audit-credential"
        body = ("x" * 2035 + secret + " tail").encode()
        error = urllib.error.HTTPError("https://example.test/models", 401, "denied", {}, io.BytesIO(body))
        with mock.patch.dict(os.environ, {"VENICE_API_KEY": "test-fake-key"}), \
             mock.patch("venice.client.urllib.request.urlopen", side_effect=error):
            with self.assertRaises(VeniceAPIError) as caught:
                VeniceClient(secret).get_json("/models")
        self.assertNotIn(secret, str(caught.exception))
        self.assertNotIn(secret[:10], caught.exception.body)
        self.assertIn("[REDACTED]", caught.exception.body)

    def test_http_error_code_and_transport_reason_do_not_echo_key(self):
        secret = "synthetic-error-code-credential"
        client = VeniceClient(secret)
        with self.assertRaises(VeniceAPIError) as caught:
            client._raise_api_error(401, "https://example.test/models",
                                    json.dumps({"code": secret}).encode(), "application/json")
        self.assertNotIn(secret, str(caught.exception))
        self.assertNotIn(secret, caught.exception.code)
        with mock.patch("venice.client.urllib.request.urlopen",
                        side_effect=urllib.error.URLError(secret)):
            with self.assertRaises(VeniceAPIError) as caught:
                client.get_json("/models")
        self.assertNotIn(secret, str(caught.exception))

    def test_sdk_diagnostic_redacts_request_bearer_even_without_matching_env(self):
        class SDKError(Exception):
            request = SimpleNamespace(headers={"authorization": "Bearer synthetic-sdk-credential"})

        error = SDKError("rejected synthetic-sdk-credential")
        self.assertEqual(_openai._safe_api_error(error, 401), "rejected [REDACTED]")

    def test_environment_secret_is_removed_from_generic_api_error(self):
        with mock.patch.dict(os.environ, {"VENICE_API_KEY": "synthetic-env-credential"}):
            error = VeniceAPIError(401, "/models", "rejected synthetic-env-credential")
        self.assertNotIn("synthetic-env-credential", str(error))
        self.assertNotIn("synthetic-env-credential", error.body)
