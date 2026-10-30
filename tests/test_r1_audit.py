"""r1 audit regressions: video cleanup wire contract and credential redaction.

Test author of record: Claude (production change 1a844f2 is Codex's).

Only the Venice HTTP peer is faked -- by a loopback server (tests/_r1_loopback.py)
for everything that can reach a socket, and by patching ``urlopen`` solely where
a transport *exception* (URLError/OSError) must be produced. The CLI runs as a
real subprocess with a temporary HOME; the MCP handlers, ``VeniceClient``,
``VeniceAPIError``, ``_redact`` and ``_openai._safe_api_error`` all run unmodified.
Every credential below is synthetic; none is a real key and none is read from disk.
"""
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
import urllib.error
import urllib.parse
from contextlib import redirect_stderr
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import venice
from tests._r1_loopback import (
    MP3, MP4, LoopbackVenice, Reply, bytes_reply, echo_bearer, json_reply,
    model_catalog, video_model,
)
from venice import _redact, config
from venice.client import VeniceAPIError, VeniceClient
from venice.commands import _mcp, _openai, _video_jobs, video

SRC = Path(venice.__file__).resolve().parents[1]  # follows PYTHONPATH (broken copies)

# Synthetic credentials. Each contains characters that differ between the raw,
# JSON-escaped, %-quoted, and quote_plus forms, so every variant is distinct.
KEY = 'r1Zq9/SYNTH+cred "7Hk2" end'
CLIENT_KEY = "r1-file-loaded-synthetic-cred-Qm4"
EXCERPT = 2048
FRAGMENT = 6


def variants(secret):
    """The encodings a hostile upstream might echo, computed independently."""
    return {
        "raw": secret,
        "json": json.dumps(secret)[1:-1],
        "quote": urllib.parse.quote(secret, safe=""),
        "quote_plus": urllib.parse.quote_plus(secret, safe=""),
    }


class _Env(unittest.TestCase):
    """Temp HOME, temp job registry, no ambient VENICE_API_KEY unless set."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        env = mock.patch.dict(os.environ, {
            "HOME": str(self.tmp),
            "XDG_CONFIG_HOME": str(self.tmp / ".config"),
            config.ENV_VIDEO_JOBS_FILE: str(self.tmp / "jobs.json"),
        })
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop("VENICE_API_KEY", None)

    def set_env_key(self, value):
        os.environ["VENICE_API_KEY"] = value

    def assert_no_fragment(self, text, secret, n=FRAGMENT):
        """No n-character window of ``secret`` (any variant) appears in ``text``."""
        text = str(text)
        for name, form in variants(secret).items():
            for i in range(0, max(1, len(form) - n + 1)):
                window = form[i:i + n]
                self.assertNotIn(window, text, f"{name} fragment {window!r} leaked")


# --------------------------------------------------------------------------- #
# Video cleanup wire contract
# --------------------------------------------------------------------------- #

class CompleteBodyContract(unittest.TestCase):
    def test_complete_body_is_exactly_queue_id(self):
        for queue_id in ("job", "", "üñí-кю", "q" * 4096, "has space/slash"):
            with self.subTest(queue_id=queue_id[:12]):
                self.assertEqual(video.complete_body(queue_id), {"queue_id": queue_id})

    def test_complete_body_cannot_be_given_a_model(self):
        with self.assertRaises(TypeError):
            video.complete_body("model", "job")

    def test_complete_body_returns_independent_dicts(self):
        first = video.complete_body("a")
        first["model"] = "leak"
        self.assertEqual(video.complete_body("a"), {"queue_id": "a"})

    def test_retrieve_body_still_carries_model(self):
        self.assertEqual(video.job_body("m-1", "q-1"), {"model": "m-1", "queue_id": "q-1"})
        self.assertEqual(set(video.job_body("", "")), {"model", "queue_id"})


class CliVideoStatusCleanup(_Env):
    """`venice video-status` as a subprocess against the loopback API."""

    def run_cli(self, api, *argv, key="test-fake-key"):
        env = {
            "PATH": os.environ.get("PATH", ""),
            "HOME": str(self.tmp),
            "XDG_CONFIG_HOME": str(self.tmp / ".config"),
            "VENICE_API_KEY": key,
            "VENICE_BASE_URL": api.base_url,
            config.ENV_VIDEO_JOBS_FILE: str(self.tmp / "jobs.json"),
            "PYTHONPATH": str(SRC),
        }
        return subprocess.run(
            [sys.executable, "-m", "venice", "video-status", *argv,
             "--poll-interval", "0", "--max-wait", "10"],
            cwd=str(self.tmp), env=env, capture_output=True, text=True, timeout=60,
        )

    def test_cleanup_posts_only_queue_id_after_saving(self):
        out = self.tmp / "clip.mp4"
        with LoopbackVenice() as api:
            api.on("POST", "/video/retrieve",
                   json_reply({"status": "PROCESSING"}), bytes_reply(MP4, "video/mp4"))
            api.on("POST", "/video/complete", json_reply({"success": True}))
            proc = self.run_cli(api, "job-77", "--model", "r1-video", "-o", str(out))
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(api.unexpected, [])
        self.assertEqual(out.read_bytes(), MP4)
        self.assertEqual(api.paths(), ["/video/retrieve", "/video/retrieve", "/video/complete"])
        self.assertEqual(api.bodies("/video/retrieve"),
                         [{"model": "r1-video", "queue_id": "job-77"}] * 2)
        self.assertEqual(api.bodies("/video/complete"), [{"queue_id": "job-77"}])
        self.assertTrue(all(s.authorized for s in api.seen))

    def test_no_cleanup_flag_skips_complete(self):
        out = self.tmp / "clip.mp4"
        with LoopbackVenice() as api:
            api.on("POST", "/video/retrieve", bytes_reply(MP4, "video/mp4"))
            proc = self.run_cli(api, "job-8", "--model", "r1-video", "-o", str(out),
                                "--no-cleanup")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(api.paths(), ["/video/retrieve"])
        self.assertEqual(api.unexpected, [])

    def test_failed_cleanup_warns_without_echoing_key(self):
        out = self.tmp / "clip.mp4"
        key = "r1-cli-synthetic-credential-Zx81"
        with LoopbackVenice() as api:
            api.on("POST", "/video/retrieve", bytes_reply(MP4, "video/mp4"))
            api.on("POST", "/video/complete", echo_bearer(
                422, '{"error": {"code": "{key}", "message": "bad token {key}"}}'))
            proc = self.run_cli(api, "job-9", "--model", "r1-video", "-o", str(out), key=key)
        self.assertEqual(proc.returncode, 0, proc.stderr)  # cleanup is best-effort
        self.assertEqual(out.read_bytes(), MP4)
        self.assertIn("warning: cleanup call failed", proc.stderr)
        self.assertIn("HTTP 422", proc.stderr)
        self.assertIn("[REDACTED]", proc.stderr)
        self.assert_no_fragment(proc.stderr + proc.stdout, key)
        self.assertEqual(api.bodies("/video/complete"), [{"queue_id": "job-9"}])


class McpCleanupDelegation(_Env):
    """The print-free MCP handlers, with a real client over loopback."""

    def client(self, api, key="test-fake-key"):
        return VeniceClient(key, base_url=api.base_url)

    def test_video_tool_end_to_end_cleanup_body(self):
        with LoopbackVenice() as api:
            api.on("GET", "/models", model_catalog({"video": [video_model()]}))
            api.on("POST", "/video/quote", json_reply({"quote": 0.05}))
            api.on("POST", "/video/queue", json_reply({"model": "r1-video", "queue_id": "vq-1"}))
            api.on("POST", "/video/retrieve", bytes_reply(MP4, "video/mp4"))
            api.on("POST", "/video/complete", json_reply({"success": True}))
            result = _mcp.video_tool(
                self.client(api), "a koi pond", confirm=True, max_spend=1.0,
                output_dir=str(self.tmp / "out"), max_wait=5,
            )
        self.assertEqual(result["status"], "ok", result)
        self.assertEqual(Path(result["path"]).read_bytes(), MP4)
        self.assertEqual(result["queue_id"], "vq-1")
        self.assertEqual(api.unexpected, [])
        self.assertEqual(api.bodies("/video/retrieve"), [{"model": "r1-video", "queue_id": "vq-1"}])
        self.assertEqual(api.bodies("/video/complete"), [{"queue_id": "vq-1"}])
        self.assertEqual(api.paths()[-1], "/video/complete")

    def test_video_tool_survives_cleanup_failure(self):
        with LoopbackVenice() as api:
            api.on("GET", "/models", model_catalog({"video": [video_model()]}))
            api.on("POST", "/video/quote", json_reply({"quote": 0.05}))
            api.on("POST", "/video/queue", json_reply({"queue_id": "vq-2"}))
            api.on("POST", "/video/retrieve", bytes_reply(MP4, "video/mp4"))
            api.on("POST", "/video/complete", json_reply({"error": "gone"}, 500))
            result = _mcp.video_tool(
                self.client(api), "p", confirm=True, max_spend=1.0,
                output_dir=str(self.tmp / "out"), max_wait=5,
            )
        self.assertEqual(result["status"], "ok", result)
        self.assertEqual(api.bodies("/video/complete"), [{"queue_id": "vq-2"}])

    def test_complete_job_video_sends_queue_id_and_forgets_private_metadata(self):
        _video_jobs.remember("vq-3", "r1-video", "https://media.example.test/signed")
        self.assertIsNotNone(_video_jobs.lookup("vq-3", "r1-video"))
        with LoopbackVenice() as api:
            api.on("POST", "/video/complete", json_reply({"success": True}))
            self.assertIsNone(_mcp.complete_job(
                self.client(api), queue_id="vq-3", type="video", model="r1-video"))
        self.assertEqual(api.bodies("/video/complete"), [{"queue_id": "vq-3"}])
        self.assertIsNone(_video_jobs.lookup("vq-3", "r1-video"))
        self.assertEqual(api.unexpected, [])

    def test_complete_job_video_forgets_even_when_backend_rejects(self):
        _video_jobs.remember("vq-4", "r1-video", "https://media.example.test/signed")
        with LoopbackVenice() as api:
            api.on("POST", "/video/complete", json_reply({"error": "nope"}, 400))
            _mcp.complete_job(self.client(api), queue_id="vq-4", type="VIDEO", model="r1-video")
        self.assertEqual(api.bodies("/video/complete"), [{"queue_id": "vq-4"}])
        self.assertIsNone(_video_jobs.lookup("vq-4", "r1-video"))

    def test_complete_job_audio_types_keep_model(self):
        for kind in ("sfx", "music", "Music"):
            with self.subTest(type=kind), LoopbackVenice() as api:
                api.on("POST", "/audio/complete", json_reply({"success": True}))
                _mcp.complete_job(self.client(api), queue_id="aq", type=kind, model="m-a")
                self.assertEqual(api.bodies("/audio/complete"), [{"model": "m-a", "queue_id": "aq"}])
                self.assertEqual(api.unexpected, [])

    def test_complete_job_unknown_type_sends_nothing(self):
        for kind in ("image", "", None):
            with self.subTest(type=kind), LoopbackVenice() as api:
                _mcp.complete_job(self.client(api), queue_id="q", type=kind, model="m")
                self.assertEqual(api.seen, [])

    def test_job_result_video_completes_with_queue_id_only(self):
        with LoopbackVenice() as api:
            api.on("POST", "/video/retrieve", bytes_reply(MP4, "video/mp4"))
            api.on("POST", "/video/complete", json_reply({"success": True}))
            result = _mcp.job_result_tool(
                self.client(api), queue_id="vq-5", type="video", model="r1-video",
                output_dir=str(self.tmp / "out"),
            )
        self.assertEqual(result["status"], "ok", result)
        self.assertEqual(api.bodies("/video/retrieve"), [{"model": "r1-video", "queue_id": "vq-5"}])
        self.assertEqual(api.bodies("/video/complete"), [{"queue_id": "vq-5"}])

    def test_job_result_audio_completes_with_model(self):
        with LoopbackVenice() as api:
            api.on("POST", "/audio/retrieve", bytes_reply(MP3, "audio/mpeg"))
            api.on("POST", "/audio/complete", json_reply({"success": True}))
            result = _mcp.job_result_tool(
                self.client(api), queue_id="aq-5", type="sfx", model="m-s",
                output_dir=str(self.tmp / "out"),
            )
        self.assertEqual(result["status"], "ok", result)
        self.assertEqual(api.bodies("/audio/complete"), [{"model": "m-s", "queue_id": "aq-5"}])


# --------------------------------------------------------------------------- #
# _redact.redact_text
# --------------------------------------------------------------------------- #

class RedactText(_Env):
    def test_every_encoded_variant_of_an_explicit_secret_is_removed(self):
        for name, form in variants(KEY).items():
            with self.subTest(variant=name):
                self.assertEqual(_redact.redact_text(form, KEY), "[REDACTED]")
                self.assertEqual(_redact.redact_text(f"<{form}>", KEY), "<[REDACTED]>")

    def test_environment_secret_is_removed_without_being_passed(self):
        self.set_env_key(KEY)
        for name, form in variants(KEY).items():
            with self.subTest(variant=name):
                self.assertEqual(_redact.redact_text(f"a {form} b"), "a [REDACTED] b")

    def test_environment_and_explicit_secrets_both_apply(self):
        self.set_env_key("env-synthetic-secret")
        self.assertEqual(
            _redact.redact_text("env-synthetic-secret / file-synthetic-secret",
                                "file-synthetic-secret"),
            "[REDACTED] / [REDACTED]",
        )

    def test_every_occurrence_is_removed(self):
        self.assertEqual(_redact.redact_text("k1 k1 k1", "k1"), "[REDACTED] [REDACTED] [REDACTED]")

    def test_longer_secret_wins_over_its_prefix(self):
        text = "token=abcdef-long and abc"
        self.assertEqual(_redact.redact_text(text, "abc", "abcdef-long"),
                         "token=[REDACTED] and [REDACTED]")
        self.assertEqual(_redact.redact_text(text, "abcdef-long", "abc"),
                         "token=[REDACTED] and [REDACTED]")

    def test_encoded_form_containing_the_raw_form_is_removed_whole(self):
        # quote_plus("a b") = "a+b"; the secret "a+b" itself is also known. The
        # longer %-quoted form of "a+b" ("a%2Bb") must not be left half-done.
        self.assertEqual(_redact.redact_text("a%2Bb|a+b|a b", "a+b", "a b"),
                         "[REDACTED]|[REDACTED]|[REDACTED]")

    def test_slash_is_percent_encoded_in_the_quote_variant(self):
        self.assertEqual(_redact.redact_text("u=sk%2Fpart", "sk/part"), "u=[REDACTED]")
        self.assertEqual(_redact.redact_text("u=sk+part", "sk part"), "u=[REDACTED]")
        self.assertEqual(_redact.redact_text("u=sk%20part", "sk part"), "u=[REDACTED]")

    def test_json_escapes_unicode_and_controls(self):
        secret = "kÿ\n\\é"
        escaped = json.dumps(secret)[1:-1]
        self.assertIn("\\u00ff", escaped)
        self.assertEqual(_redact.redact_text(f'"{escaped}"', secret), '"[REDACTED]"')

    def test_empty_and_non_string_secrets_are_ignored(self):
        text = "nothing secret here"
        self.assertEqual(_redact.redact_text(text), text)
        self.assertEqual(_redact.redact_text(text, ""), text)
        self.assertEqual(_redact.redact_text(text, None, 7, b"here"), text)
        self.set_env_key("")
        self.assertEqual(_redact.redact_text(text), text)

    def test_non_string_values_are_stringified(self):
        self.assertEqual(_redact.redact_text(123456, "345"), "12[REDACTED]6")
        self.assertEqual(_redact.redact_text(ValueError("x sk-1 y"), "sk-1"), "x [REDACTED] y")
        self.assertEqual(_redact.redact_text(None), "None")


# --------------------------------------------------------------------------- #
# Client error path: VeniceClient -> _raise_api_error -> VeniceAPIError
# --------------------------------------------------------------------------- #

class ClientErrorRedaction(_Env):
    def fail_with(self, reply, key=CLIENT_KEY, path="/models"):
        with LoopbackVenice() as api:
            api.on("GET", path.split("?")[0], reply)
            with self.assertRaises(VeniceAPIError) as caught:
                VeniceClient(key, base_url=api.base_url).get_json(path)
        self.assertEqual(api.unexpected, [])
        return caught.exception

    def assert_clean(self, error, secret):
        for surface in (str(error), error.body, error.url, error.code or ""):
            self.assert_no_fragment(surface, secret)

    def test_echoed_bearer_is_redacted_from_every_field(self):
        error = self.fail_with(echo_bearer(
            401, '{"error": {"code": "{key}", "message": "invalid key {key}"}}'))
        self.assertEqual(error.status, 401)
        self.assertEqual(error.code, "[REDACTED]")
        self.assertIn("invalid key [REDACTED]", error.body)
        self.assertIn("[[REDACTED]]", str(error))
        self.assert_clean(error, CLIENT_KEY)

    def test_top_level_code_field_is_redacted(self):
        error = self.fail_with(echo_bearer(403, '{"code": "{key}"}'))
        self.assertEqual(error.code, "[REDACTED]")
        self.assert_clean(error, CLIENT_KEY)

    def test_file_loaded_key_is_redacted_without_any_environment_key(self):
        self.assertNotIn("VENICE_API_KEY", os.environ)
        error = self.fail_with(echo_bearer(401, "denied {key}", "text/plain"))
        self.assertEqual(error.body, "denied [REDACTED]")
        self.assert_clean(error, CLIENT_KEY)

    def test_file_loaded_key_is_redacted_when_environment_holds_another(self):
        self.set_env_key("unrelated-env-synthetic")
        error = self.fail_with(echo_bearer(401, "denied {key} unrelated-env-synthetic",
                                           "text/plain"))
        self.assertEqual(error.body, "denied [REDACTED] [REDACTED]")

    def test_encoded_echoes_are_redacted(self):
        for name, form in variants(KEY).items():
            with self.subTest(variant=name):
                error = self.fail_with(Reply(400, "text/plain", f"got <{form}>"), key=KEY)
                self.assertEqual(error.body, "got <[REDACTED]>")
                self.assert_clean(error, KEY)

    def test_html_and_invalid_utf8_bodies_are_redacted(self):
        body = b"<html>\xff\xfe" + CLIENT_KEY.encode() + b"\xc3</html>"
        error = self.fail_with(Reply(502, "text/html", body))
        self.assertIn("[REDACTED]", error.body)
        self.assertIn("�", error.body)
        self.assert_clean(error, CLIENT_KEY)

    def test_malformed_json_error_body_is_still_redacted(self):
        error = self.fail_with(Reply(500, "application/json", '{"error": "' + CLIENT_KEY))
        self.assertIsNone(error.code)
        self.assertEqual(error.body, '{"error": "[REDACTED]')

    def test_key_in_request_path_is_redacted_from_url(self):
        quoted = urllib.parse.quote(KEY, safe="")
        error = self.fail_with(json_reply({"error": "nope"}, 404), key=KEY,
                               path="/models/" + quoted)
        self.assertTrue(error.url.endswith("/models/[REDACTED]"), error.url)
        self.assert_clean(error, KEY)

    def test_empty_error_body(self):
        error = self.fail_with(Reply(503, "application/json", b""))
        self.assertEqual(error.body, "")
        self.assertIsNone(error.code)
        self.assertEqual(str(error), f"HTTP 503 from {error.url}")

    # -- the 2 KB excerpt boundary ------------------------------------------- #
    def test_excerpt_boundary_never_leaks_a_partial_key(self):
        length = len(CLIENT_KEY)
        offsets = sorted({0, 1, EXCERPT - length - 1, EXCERPT - length, EXCERPT - length + 1,
                          EXCERPT - 8, EXCERPT - 1, EXCERPT, EXCERPT + 1, 2 * EXCERPT})
        for offset in offsets:
            with self.subTest(offset=offset):
                body = "x" * offset + CLIENT_KEY + " tail"
                error = self.fail_with(Reply(401, "text/plain", body))
                expected = ("x" * offset + "[REDACTED] tail")[:EXCERPT]
                self.assertEqual(error.body, expected)
                self.assertLessEqual(len(error.body), EXCERPT)
                self.assert_clean(error, CLIENT_KEY)

    def test_excerpt_length_at_exactly_2kb_and_neighbours(self):
        for size in (EXCERPT - 1, EXCERPT, EXCERPT + 1):
            with self.subTest(size=size):
                error = self.fail_with(Reply(500, "text/plain", "y" * size))
                self.assertEqual(error.body, "y" * min(size, EXCERPT))

    def test_message_body_preview_is_capped_at_500_after_redaction(self):
        error = self.fail_with(Reply(401, "text/plain", "z" * 495 + CLIENT_KEY))
        self.assertEqual(str(error).split("\n  body: ", 1)[1], ("z" * 495 + "[REDACTED]")[:500])

    # -- transport failures (exceptions must come from the HTTP boundary) ----- #
    def test_connection_error_reason_is_redacted(self):
        client = VeniceClient(CLIENT_KEY, base_url="http://127.0.0.1:9/api/v1")
        for exc in (urllib.error.URLError(f"proxy said {CLIENT_KEY}"),
                    OSError(f"socket {CLIENT_KEY}")):
            with self.subTest(exc=type(exc).__name__), \
                 mock.patch("venice.client.urllib.request.urlopen", side_effect=exc):
                with self.assertRaises(VeniceAPIError) as caught:
                    client.get_json("/models")
                self.assertEqual(caught.exception.status, 0)
                self.assertIn("connection error: ", caught.exception.body)
                self.assertIn("[REDACTED]", caught.exception.body)
                self.assert_clean(caught.exception, CLIENT_KEY)


class VeniceAPIErrorDirect(_Env):
    def test_secret_keyword_redacts_url_body_code_and_message(self):
        error = VeniceAPIError(401, f"https://h.example.test/a/{CLIENT_KEY}?k={CLIENT_KEY}",
                               f"body {CLIENT_KEY}", code=f"C-{CLIENT_KEY}", secret=CLIENT_KEY)
        self.assertEqual(error.body, "body [REDACTED]")
        self.assertEqual(error.code, "C-[REDACTED]")
        self.assertNotIn(CLIENT_KEY, error.url)
        self.assertNotIn(CLIENT_KEY, str(error))

    def test_environment_key_redacted_without_secret_keyword(self):
        self.set_env_key(CLIENT_KEY)
        error = VeniceAPIError(401, "/models", f"rejected {CLIENT_KEY}")
        self.assertEqual(error.body, "rejected [REDACTED]")
        self.assertNotIn(CLIENT_KEY, str(error))

    def test_falsy_code_stays_none_and_is_not_rendered(self):
        for code in (None, ""):
            with self.subTest(code=code):
                error = VeniceAPIError(500, "/x", "b", code=code)
                self.assertIsNone(error.code)
                self.assertEqual(str(error), "HTTP 500 from /x\n  body: b")

    def test_non_string_code_is_rendered_as_text(self):
        error = VeniceAPIError(500, "/x", "", code=42)
        self.assertEqual(error.code, "42")
        self.assertEqual(str(error), "HTTP 500 from /x [42]")


# --------------------------------------------------------------------------- #
# SDK diagnostics and the MCP error envelope
# --------------------------------------------------------------------------- #

class _SDKError(Exception):
    """Shape of an openai SDK APIError: the message plus the httpx request."""

    def __init__(self, message, headers):
        super().__init__(message)
        self.request = SimpleNamespace(headers=headers) if headers is not None else None


class SdkDiagnostics(_Env):
    def test_bearer_from_request_is_redacted_without_environment(self):
        for prefix in ("Bearer ", "bearer ", "BEARER "):
            with self.subTest(prefix=prefix):
                error = _SDKError(f"rejected {CLIENT_KEY}", {"authorization": prefix + CLIENT_KEY})
                self.assertEqual(_openai._safe_api_error(error, 401), "rejected [REDACTED]")

    def test_encoded_bearer_echo_is_redacted(self):
        quoted = urllib.parse.quote(KEY, safe="")
        error = _SDKError(f"url had {quoted}", {"authorization": "Bearer " + KEY})
        self.assertEqual(_openai._safe_api_error(error, 400), "url had [REDACTED]")

    def test_missing_or_odd_headers_fall_back_to_environment(self):
        self.set_env_key(CLIENT_KEY)
        for headers in (None, {}, {"authorization": b"Bearer x"}, {"authorization": "Bearer "}):
            with self.subTest(headers=headers):
                error = _SDKError(f"oops {CLIENT_KEY}", headers)
                self.assertEqual(_openai._safe_api_error(error, 500), "oops [REDACTED]")

    def test_empty_bearer_does_not_mangle_text(self):
        error = _SDKError("plain message", {"authorization": "Bearer "})
        self.assertEqual(_openai._safe_api_error(error, 500), "plain message")

    def test_redaction_happens_before_truncation(self):
        error = _SDKError("p" * 990 + CLIENT_KEY, {"authorization": "Bearer " + CLIENT_KEY})
        text = _openai._safe_api_error(error, 500)
        self.assertEqual(text, ("p" * 990 + "[REDACTED]"))
        long_error = _SDKError("p" * 995 + CLIENT_KEY, {"authorization": "Bearer " + CLIENT_KEY})
        text = _openai._safe_api_error(long_error, 500)
        self.assertEqual(text, "p" * 995 + "[R...")
        self.assert_no_fragment(text, CLIENT_KEY)

    def test_html_bodies_collapse_to_title(self):
        error = _SDKError(f"<html><body>{CLIENT_KEY}</body></html>",
                          {"authorization": "Bearer " + CLIENT_KEY})
        self.assertEqual(_openai._safe_api_error(error, 413), "HTTP 413 Request Entity Too Large")

    def test_status_to_exit_connection_error_is_redacted(self):
        class APIConnectionError(Exception):
            pass

        module = SimpleNamespace(APIConnectionError=APIConnectionError)
        error = APIConnectionError(f"connect failed {CLIENT_KEY}")
        error.request = SimpleNamespace(headers={"authorization": "Bearer " + CLIENT_KEY})
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            rc = _openai.status_to_exit(module, error, "chat")
        self.assertEqual(rc, 8)
        self.assertEqual(stderr.getvalue(), "chat: connection error: connect failed [REDACTED]\n")


class McpErrorEnvelope(_Env):
    def test_err_redacts_environment_key(self):
        self.set_env_key(KEY)
        for form in variants(KEY).values():
            self.assertEqual(_mcp._err(f"boom {form}"),
                             {"status": "error", "message": "boom [REDACTED]"})

    def test_reflected_argument_is_redacted_in_tool_error(self):
        # job_status echoes the host-supplied type via repr(); a host that was
        # tricked into sending the key back must not get it printed into logs.
        self.set_env_key(KEY)
        with LoopbackVenice() as api:
            result = _mcp.job_status_tool(VeniceClient("x", base_url=api.base_url),
                                          queue_id="q", type=KEY, model="m")
        self.assertEqual(result["status"], "error")
        self.assertIn("[REDACTED]", result["message"])
        self.assert_no_fragment(result["message"], KEY)
        self.assertEqual(api.seen, [])

    def test_upstream_echo_through_video_tool_is_redacted(self):
        with LoopbackVenice() as api:
            api.on("GET", "/models", model_catalog({"video": [video_model()]}))
            api.on("POST", "/video/quote", echo_bearer(
                402, '{"error": {"code": "{key}", "message": "balance for {key}"}}'))
            result = _mcp.video_tool(VeniceClient(CLIENT_KEY, base_url=api.base_url),
                                     "p", confirm=True, output_dir=str(self.tmp))
        self.assertEqual(result["status"], "error")
        self.assertTrue(result["message"].startswith("video quote rejected: HTTP 402"), result)
        self.assert_no_fragment(result["message"], CLIENT_KEY)
        self.assertNotIn("/video/queue", api.paths())

    def test_upstream_echo_through_job_status_is_redacted(self):
        with LoopbackVenice() as api:
            api.on("POST", "/video/retrieve", echo_bearer(500, "internal {key}", "text/plain"))
            result = _mcp.job_status_tool(VeniceClient(CLIENT_KEY, base_url=api.base_url),
                                          queue_id="q", type="video", model="m")
        self.assertEqual(result["status"], "error")
        self.assertIn("internal [REDACTED]", result["message"])
        self.assert_no_fragment(json.dumps(result), CLIENT_KEY)


if __name__ == "__main__":
    unittest.main()
