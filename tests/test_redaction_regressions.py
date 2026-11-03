"""Adversarial diagnostics using synthetic, already-loaded credentials only."""
import io
import json
import os
import re
import sys
import unittest
import urllib.error
import urllib.parse
from types import SimpleNamespace
from unittest import mock

from tests.test_client import FakeResp
from venice import _redact
from venice.client import VeniceAPIError, VeniceClient
from venice.commands import _mcp, _openai, _shared


class RedactionRegressions(unittest.TestCase):
    def setUp(self):
        # No test may resolve ambient credentials, even on an error path.
        guard = mock.patch("venice.auth.load_key", side_effect=AssertionError("unexpected auth read"))
        guard.start()
        self.addCleanup(guard.stop)
        env = mock.patch.dict(os.environ, {"VENICE_API_KEY": "test-fake-key"})
        env.start()
        self.addCleanup(env.stop)

    def test_lowercase_percent_escapes(self):
        secret = "synthetic/percent+credential"
        encoded = "synthetic%2fpercent%2bcredential"
        self.assertEqual(_redact.redact_text(f"denied {encoded}", secret), "denied [REDACTED]")

    def test_mixed_case_percent_escapes(self):
        secret = "synthetic/percent+credential/again"
        encoded = "synthetic%2fpercent%2Bcredential%2Fagain"
        self.assertEqual(_redact.redact_text(encoded, secret), "[REDACTED]")

    def test_percent_matching_preserves_literal_case(self):
        self.assertEqual(_redact.redact_text("synthetic%2fCase", "synthetic/Case"), "[REDACTED]")
        self.assertEqual(_redact.redact_text("synthetic%2fcase", "synthetic/Case"), "synthetic%2fcase")

    def test_json_escaped_slashes(self):
        secret = 'synthetic/json/"credential\\suffix'
        encoded = r'synthetic\/json\/\"credential\\suffix'
        self.assertEqual(_redact.redact_text(f"denied {encoded}", secret), "denied [REDACTED]")

    def test_environment_key_in_both_new_encodings(self):
        with mock.patch.dict(os.environ, {"VENICE_API_KEY": "synthetic/env/credential"}):
            for encoded in ("synthetic%2fenv%2fcredential", r"synthetic\/env\/credential"):
                with self.subTest(encoded=encoded):
                    self.assertEqual(_redact.redact_text(encoded), "[REDACTED]")

    def test_url_plus_and_unicode_encodings(self):
        secret = "synthetic /caf\u00e9+credential"
        for quote in (urllib.parse.quote, urllib.parse.quote_plus):
            encoded = re.sub(r"%[0-9A-F]{2}", lambda m: m[0].lower(), quote(secret, safe=""))
            with self.subTest(quote=quote.__name__):
                self.assertEqual(_redact.redact_text(encoded, secret), "[REDACTED]")

    def test_existing_encodings_and_unrelated_text_are_preserved(self):
        secret = 'synthetic/"credential'
        for encoded in (secret, json.dumps(secret)[1:-1], urllib.parse.quote(secret, safe="")):
            with self.subTest(encoded=encoded):
                self.assertEqual(_redact.redact_text(f"before {encoded} after", secret), "before [REDACTED] after")
        self.assertEqual(_redact.redact_text("ordinary / path %2f", "", None), "ordinary / path %2f")

    def test_regex_characters_and_overlapping_keys_are_literal(self):
        secret = "synthetic/.*[credential](value)+"
        self.assertEqual(_redact.redact_text(secret, secret), "[REDACTED]")
        self.assertEqual(_redact.redact_text("synthetic/abc", secret), "synthetic/abc")
        self.assertEqual(_redact.redact_text("long-credential", "credential", "long-credential"), "[REDACTED]")
        self.assertEqual(_redact.redact_text("credential", "credential", "REDACTED"), "[REDACTED]")

    def test_tool_validation_redacts_already_loaded_client_key(self):
        secret = "synthetic-file-origin-credential"
        result = _mcp.sfx_tool(VeniceClient(secret), "a sound", model=secret)
        self.assertEqual(result["status"], "error")
        self.assertIn("unknown model", result["message"])
        self.assertNotIn(secret, result["message"])
        self.assertIn("[REDACTED]", result["message"])

    def test_nested_media_validation_redacts_client_key(self):
        secret = "synthetic-file-origin-credential"
        authority = SimpleNamespace(resolve=mock.Mock(side_effect=_shared.MediaPathError(f"denied {secret}")))
        result = _mcp.upscale_tool(VeniceClient(secret), "fixture.png", path_authority=authority)
        self.assertEqual(result["status"], "error")
        self.assertNotIn(secret, result["message"])
        self.assertIn("[REDACTED]", result["message"])

    def test_sdk_tool_error_redacts_client_key_before_truncation(self):
        secret = "synthetic-file-origin-credential"
        sdk_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
            create=mock.Mock(side_effect=RuntimeError("x" * 990 + secret + " tail")),
        )))
        sdk = SimpleNamespace(OpenAIError=RuntimeError, OpenAI=mock.Mock(return_value=sdk_client))
        catalog = json.dumps({"data": [{"id": "fixture-model", "model_spec": {"supportsVision": True}}]}).encode()
        for tool in ("chat", "vision"):
            with self.subTest(tool=tool), mock.patch.object(_openai, "import_openai", return_value=sdk), \
                    mock.patch("venice.client.urllib.request.urlopen", return_value=FakeResp(200, catalog)):
                client = VeniceClient(secret)
                if tool == "chat":
                    result = _mcp.chat_tool(client, "hello", model="fixture-model")
                else:
                    result = _mcp.vision_tool(client, image_url="https://example.test/image.png", model="fixture-model")
                self.assertEqual(result["status"], "error")
                self.assertNotIn(secret[:8], result["message"])
                self.assertIn("[REDACT", result["message"])

    def test_http_error_encodings_redacted_before_excerpt(self):
        secret = "synthetic/file/credential"
        for encoded in ("synthetic%2ffile%2fcredential", r"synthetic\/file\/credential"):
            body = ("x" * 2035 + encoded + " tail").encode()
            error = urllib.error.HTTPError("https://example.test/models", 401, "denied", {}, io.BytesIO(body))
            with self.subTest(encoded=encoded), mock.patch("venice.client.urllib.request.urlopen", side_effect=error), \
                    self.assertRaises(VeniceAPIError) as caught:
                VeniceClient(secret).get_json("/models")
            self.assertNotIn("synthetic", caught.exception.body)
            self.assertIn("[REDACTED]", caught.exception.body)

    def test_transport_error_encodings_redact_client_key(self):
        secret = "synthetic/file/credential"
        for error in (urllib.error.URLError("synthetic%2ffile%2fcredential"), OSError(r"synthetic\/file\/credential")):
            with self.subTest(error=type(error).__name__), \
                    mock.patch("venice.client.urllib.request.urlopen", side_effect=error), \
                    self.assertRaises(VeniceAPIError) as caught:
                VeniceClient(secret).get_json("/models")
            self.assertNotIn("synthetic", str(caught.exception))
            self.assertIn("[REDACTED]", str(caught.exception))

    def test_sdk_connection_exit_redacts_retained_request_key(self):
        class ConnectionError(Exception):
            request = SimpleNamespace(headers={"authorization": "Bearer synthetic/sdk/credential"})

        error = ConnectionError(r"synthetic\/sdk\/credential")
        output = io.StringIO()
        with mock.patch.object(sys, "stderr", output):
            code = _openai.status_to_exit(SimpleNamespace(APIConnectionError=ConnectionError), error, "chat")
        self.assertEqual(code, 8)
        self.assertEqual(output.getvalue(), "chat: connection error: [REDACTED]\n")


if __name__ == "__main__":
    unittest.main()
