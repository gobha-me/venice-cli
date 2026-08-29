"""Unit tests for VeniceClient. Mocks urllib.request.urlopen."""
import io
import json
import socket
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock
from urllib.error import HTTPError, URLError

from venice import __version__
from venice.client import VeniceAPIError, VeniceClient


class FakeResp:
    def __init__(self, status=200, body=b"", ctype="application/json"):
        self.status = status
        self._body = body
        self._offset = 0
        self.headers = {"Content-Type": ctype}

    def read(self, n=-1):
        if n is None or n < 0:
            result = self._body[self._offset:]
            self._offset = len(self._body)
            return result
        result = self._body[self._offset:self._offset + n]
        self._offset += len(result)
        return result

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class TestVeniceClient(unittest.TestCase):

    def test_post_json_sets_auth_header_and_encodes_body(self):
        c = VeniceClient(api_key="test-fake-key")
        captured = {}

        def fake_urlopen(req, timeout=None):
            captured["url"] = req.full_url
            captured["headers"] = dict(req.header_items())
            captured["body"] = req.data
            captured["method"] = req.get_method()
            return FakeResp(200, b'{"ok": true}')

        with mock.patch("venice.client.urllib.request.urlopen", fake_urlopen):
            out = c.post_json("/audio/quote", {"model": "x", "duration_seconds": 5})

        self.assertEqual(out, {"ok": True})
        self.assertEqual(captured["method"], "POST")
        self.assertTrue(captured["url"].endswith("/audio/quote"))
        h = {k.lower(): v for k, v in captured["headers"].items()}
        self.assertEqual(h["authorization"], "Bearer test-fake-key")
        self.assertEqual(h["content-type"], "application/json")
        self.assertEqual(h["user-agent"], f"venice-cli/{__version__}")
        self.assertEqual(
            json.loads(captured["body"]),
            {"model": "x", "duration_seconds": 5},
        )

    def test_download_streams_to_temp_without_authorization(self):
        c = VeniceClient(api_key="test-fake-key")
        captured = {}

        class Opener:
            def open(self, req, timeout=None):
                captured["headers"] = dict(req.header_items())
                captured["timeout"] = timeout
                return FakeResp(200, b"video", "video/mp4")

        with tempfile.TemporaryDirectory() as td, mock.patch(
            "venice.client._egress.build_https_opener", return_value=Opener()
        ):
            ctype, path, size = c.download_url_to_temp(
                "https://example.test/video.mp4?signature=secret", Path(td)
            )
            self.assertEqual(path.read_bytes(), b"video")
            self.assertEqual(size, 5)

        headers = {k.lower(): v for k, v in captured["headers"].items()}
        self.assertEqual(headers["user-agent"], f"venice-cli/{__version__}")
        self.assertNotIn("authorization", headers)
        self.assertEqual(ctype, "video/mp4")
        self.assertEqual(captured["timeout"], 60)

    def test_download_rejects_non_video_content_without_creating_a_file(self):
        c = VeniceClient(api_key="test-fake-key")

        class Opener:
            def open(self, req, timeout=None):
                return FakeResp(200, b"not video", "text/plain")

        with tempfile.TemporaryDirectory() as td, mock.patch(
            "venice.client._egress.build_https_opener", return_value=Opener()
        ):
            with self.assertRaises(VeniceAPIError):
                c.download_url_to_temp("https://example.test/video", Path(td))
            self.assertEqual(list(Path(td).iterdir()), [])

    def test_download_rejects_file_scheme_before_constructing_an_opener(self):
        c = VeniceClient(api_key="test-fake-key")
        with tempfile.TemporaryDirectory() as td, mock.patch(
            "venice.client._egress.build_https_opener"
        ) as build:
            with self.assertRaises(VeniceAPIError):
                c.download_url_to_temp("file:///tmp/harmless", Path(td))
        build.assert_not_called()

    def test_download_rejects_declared_oversize_and_empty_bodies(self):
        c = VeniceClient(api_key="test-fake-key")
        oversized = FakeResp(200, b"x", "video/mp4")
        oversized.headers["Content-Length"] = "999"

        class OversizedOpener:
            def open(self, req, timeout=None):
                return oversized

        class EmptyOpener:
            def open(self, req, timeout=None):
                return FakeResp(200, b"", "video/mp4")

        with tempfile.TemporaryDirectory() as td:
            for opener in (OversizedOpener(), EmptyOpener()):
                with self.subTest(opener=type(opener).__name__), mock.patch(
                    "venice.client._egress.build_https_opener", return_value=opener
                ), self.assertRaises(VeniceAPIError):
                    c.download_url_to_temp(
                        "https://example.test/video?signature=secret",
                        Path(td),
                        max_bytes=4,
                    )
                self.assertEqual(list(Path(td).iterdir()), [])

    def test_download_enforces_streamed_byte_limit_and_cleans_partial_file(self):
        c = VeniceClient(api_key="test-fake-key")

        class Opener:
            def open(self, req, timeout=None):
                return FakeResp(200, b"12345", "video/mp4")

        with tempfile.TemporaryDirectory() as td, mock.patch(
            "venice.client._egress.build_https_opener", return_value=Opener()
        ):
            with self.assertRaises(VeniceAPIError):
                c.download_url_to_temp(
                    "https://example.test/video?signature=secret", Path(td), max_bytes=4
                )
            self.assertEqual(list(Path(td).iterdir()), [])

    def test_download_enforces_total_deadline_and_cleans_partial_file(self):
        c = VeniceClient(api_key="test-fake-key")

        class Opener:
            def open(self, req, timeout=None):
                return FakeResp(200, b"video", "video/mp4")

        with tempfile.TemporaryDirectory() as td, mock.patch(
            "venice.client._egress.build_https_opener", return_value=Opener()
        ), mock.patch("venice.client.time.monotonic", side_effect=[0.0, 2.0]):
            with self.assertRaises(VeniceAPIError) as cm:
                c.download_url_to_temp(
                    "https://example.test/video?signature=secret",
                    Path(td),
                    max_seconds=1,
                )
            self.assertIn("timed out", str(cm.exception))
            self.assertEqual(list(Path(td).iterdir()), [])

    def test_download_open_timeout_and_oserror_become_api_errors(self):
        c = VeniceClient(api_key="test-fake-key")

        class Opener:
            def __init__(self, error):
                self.error = error

            def open(self, req, timeout=None):
                raise self.error

        with tempfile.TemporaryDirectory() as td:
            for error, text in (
                (socket.timeout("slow"), "timed out"),
                (ConnectionResetError("reset"), "connection failed"),
            ):
                with self.subTest(error=type(error).__name__), mock.patch(
                    "venice.client._egress.build_https_opener",
                    return_value=Opener(error),
                ), self.assertRaises(VeniceAPIError) as cm:
                    c.download_url_to_temp(
                        "https://example.test/video", Path(td)
                    )
                self.assertEqual(cm.exception.status, 0)
                self.assertIn(text, str(cm.exception))
                self.assertEqual(list(Path(td).iterdir()), [])

    def test_download_read_timeout_cleans_partial_file(self):
        c = VeniceClient(api_key="test-fake-key")

        class SlowResp(FakeResp):
            def read(self, n=-1):
                raise socket.timeout("slow body")

        class Opener:
            def open(self, req, timeout=None):
                return SlowResp(200, b"", "video/mp4")

        with tempfile.TemporaryDirectory() as td, mock.patch(
            "venice.client._egress.build_https_opener", return_value=Opener()
        ):
            with self.assertRaisesRegex(VeniceAPIError, "timed out"):
                c.download_url_to_temp(
                    "https://example.test/video", Path(td)
                )
            self.assertEqual(list(Path(td).iterdir()), [])

    def test_download_error_redacts_userinfo_query_and_fragment(self):
        c = VeniceClient(api_key="test-fake-key")
        url = "https://user:pass@example.test/video?signature=secret#fragment"
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(VeniceAPIError) as cm:
                c.download_url_to_temp(url, Path(td))
        rendered = str(cm.exception)
        for secret in ("user:pass", "signature", "secret", "fragment"):
            self.assertNotIn(secret, rendered)
        self.assertEqual(cm.exception.url, "https://example.test/video")

    def test_custom_user_agent_overrides_versioned_default(self):
        c = VeniceClient(api_key="test-fake-key", user_agent="embedder/2.0")
        captured = {}

        def fake_urlopen(req, timeout=None):
            captured["headers"] = dict(req.header_items())
            return FakeResp(200, b'{"ok": true}')

        with mock.patch("venice.client.urllib.request.urlopen", fake_urlopen):
            c.get_json("/models")

        headers = {k.lower(): v for k, v in captured["headers"].items()}
        self.assertEqual(headers["user-agent"], "embedder/2.0")

    def test_authorization_is_an_unredirected_header(self):
        c = VeniceClient(api_key="test-fake-key")
        captured = {}

        def fake_urlopen(req, timeout=None):
            captured["headers"] = dict(req.headers)
            captured["unredirected_headers"] = dict(req.unredirected_hdrs)
            return FakeResp(200, b'{"ok": true}')

        with mock.patch("venice.client.urllib.request.urlopen", fake_urlopen):
            c.get_json("/models")

        headers = {k.lower(): v for k, v in captured["headers"].items()}
        unredirected = {
            k.lower(): v for k, v in captured["unredirected_headers"].items()
        }
        self.assertNotIn("authorization", headers)
        self.assertEqual(
            unredirected["authorization"], "Bearer test-fake-key"
        )

    def test_cross_origin_redirect_does_not_forward_authorization(self):
        source_auth = []
        target_auth = []

        class TargetHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                target_auth.append("Authorization" in self.headers)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"ok": true}')

            def log_message(self, format, *args):
                pass

        with ThreadingHTTPServer(("127.0.0.1", 0), TargetHandler) as target:
            target_thread = threading.Thread(target=target.serve_forever)
            target_thread.start()
            target_url = f"http://127.0.0.1:{target.server_port}/target"

            class SourceHandler(BaseHTTPRequestHandler):
                def do_GET(self):
                    source_auth.append("Authorization" in self.headers)
                    self.send_response(302)
                    self.send_header("Location", target_url)
                    self.end_headers()

                def log_message(self, format, *args):
                    pass

            try:
                with ThreadingHTTPServer(("127.0.0.1", 0), SourceHandler) as source:
                    source_thread = threading.Thread(target=source.serve_forever)
                    source_thread.start()
                    try:
                        c = VeniceClient(
                            api_key="test-fake-key",
                            base_url=f"http://127.0.0.1:{source.server_port}",
                        )
                        self.assertEqual(c.get_json("/redirect"), {"ok": True})
                    finally:
                        source.shutdown()
                        source_thread.join()
            finally:
                target.shutdown()
                target_thread.join()

        self.assertEqual(source_auth, [True])
        self.assertEqual(target_auth, [False])

    def test_same_origin_redirect_does_not_forward_authorization(self):
        auth_presence = []

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                auth_presence.append("Authorization" in self.headers)
                if self.path == "/redirect":
                    self.send_response(302)
                    self.send_header("Location", "/target")
                    self.end_headers()
                    return
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"ok": true}')

            def log_message(self, format, *args):
                pass

        with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
            thread = threading.Thread(target=server.serve_forever)
            thread.start()
            try:
                c = VeniceClient(
                    api_key="test-fake-key",
                    base_url=f"http://127.0.0.1:{server.server_port}",
                )
                self.assertEqual(c.get_json("/redirect"), {"ok": True})
            finally:
                server.shutdown()
                thread.join()

        self.assertEqual(auth_presence, [True, False])

    def test_post_for_bytes_or_json_returns_bytes_on_audio_ctype(self):
        c = VeniceClient(api_key="k")
        with mock.patch(
            "venice.client.urllib.request.urlopen",
            lambda *a, **kw: FakeResp(200, b"\xff\xfbID3...", "audio/mpeg"),
        ):
            ct, payload = c.post_for_bytes_or_json(
                "/audio/retrieve", {"model": "m", "queue_id": "q"}
            )
        self.assertEqual(ct, "audio/mpeg")
        self.assertIsInstance(payload, bytes)
        self.assertTrue(payload.startswith(b"\xff\xfb"))

    def test_post_for_bytes_or_json_returns_dict_on_json_ctype(self):
        c = VeniceClient(api_key="k")
        body = json.dumps({"status": "PROCESSING"}).encode()
        with mock.patch(
            "venice.client.urllib.request.urlopen",
            lambda *a, **kw: FakeResp(200, body, "application/json"),
        ):
            ct, payload = c.post_for_bytes_or_json(
                "/audio/retrieve", {"model": "m", "queue_id": "q"}
            )
        self.assertTrue(ct.startswith("application/json"))
        self.assertEqual(payload, {"status": "PROCESSING"})

    def test_post_for_bytes_or_json_accepts_structured_json_type(self):
        c = VeniceClient(api_key="k")
        body = b'{"status":"PROCESSING"}'
        with mock.patch(
            "venice.client.urllib.request.urlopen",
            lambda *a, **kw: FakeResp(
                200, body, "application/vnd.venice.status+json; charset=utf-8"
            ),
        ):
            _ct, payload = c.post_for_bytes_or_json("/audio/retrieve", {})
        self.assertEqual(payload, {"status": "PROCESSING"})

    def test_async_malformed_and_empty_bodies_become_api_errors(self):
        c = VeniceClient(api_key="k")
        cases = (
            (b"{broken", "application/json", "non-JSON response"),
            (b"", "application/json", "empty async response"),
            (b"", "audio/mpeg", "empty async response"),
        )
        for body, ctype, message in cases:
            with self.subTest(body=body, ctype=ctype), mock.patch(
                "venice.client.urllib.request.urlopen",
                lambda *a, _body=body, _ctype=ctype, **kw: FakeResp(
                    200, _body, _ctype
                ),
            ), self.assertRaisesRegex(VeniceAPIError, message) as cm:
                c.post_for_bytes_or_json("/audio/retrieve", {})
            self.assertEqual(cm.exception.status, 200)

    def test_async_missing_or_misdeclared_json_is_decoded_conservatively(self):
        c = VeniceClient(api_key="k")
        body = b'{"status":"PROCESSING"}'
        for ctype in ("", "text/plain"):
            with self.subTest(ctype=ctype), mock.patch(
                "venice.client.urllib.request.urlopen",
                lambda *a, _ctype=ctype, **kw: FakeResp(200, body, _ctype),
            ):
                _ct, payload = c.post_for_bytes_or_json(
                    "/audio/retrieve", {}
                )
            self.assertEqual(payload, {"status": "PROCESSING"})

    def test_async_unknown_binary_type_fails_closed(self):
        c = VeniceClient(api_key="k")
        with mock.patch(
            "venice.client.urllib.request.urlopen",
            lambda *a, **kw: FakeResp(200, b"binary", "application/octet-stream"),
        ), self.assertRaisesRegex(VeniceAPIError, "unexpected async content type"):
            c.post_for_bytes_or_json("/audio/retrieve", {})

    def test_http_error_becomes_venice_api_error_with_code(self):
        c = VeniceClient(api_key="k")
        err_body = json.dumps({"code": "INSUFFICIENT_BALANCE", "message": "broke"}).encode()

        def boom(*a, **kw):
            raise HTTPError(
                url="https://api.venice.ai/api/v1/audio/queue",
                code=402,
                msg="Payment Required",
                hdrs={"Content-Type": "application/json"},  # type: ignore[arg-type]
                fp=io.BytesIO(err_body),
            )

        with mock.patch("venice.client.urllib.request.urlopen", boom):
            with self.assertRaises(VeniceAPIError) as cm:
                c.post_json("/audio/queue", {})
        self.assertEqual(cm.exception.status, 402)
        self.assertEqual(cm.exception.code, "INSUFFICIENT_BALANCE")

    def test_structured_json_http_error_extracts_code(self):
        c = VeniceClient(api_key="k")

        def boom(*a, **kw):
            raise HTTPError(
                url="https://api.venice.ai/api/v1/audio/queue",
                code=409,
                msg="Conflict",
                hdrs={"Content-Type": "application/problem+json"},  # type: ignore[arg-type]
                fp=io.BytesIO(b'{"code":"JOB_CONFLICT"}'),
            )

        with mock.patch("venice.client.urllib.request.urlopen", boom), \
             self.assertRaises(VeniceAPIError) as cm:
            c.post_json("/audio/queue", {})
        self.assertEqual(cm.exception.code, "JOB_CONFLICT")

    def test_url_error_becomes_venice_api_error_status_zero(self):
        c = VeniceClient(api_key="k")

        def boom(*a, **kw):
            raise URLError("name resolution failed")

        with mock.patch("venice.client.urllib.request.urlopen", boom):
            with self.assertRaises(VeniceAPIError) as cm:
                c.post_json("/whatever", {})
        self.assertEqual(cm.exception.status, 0)

    def test_direct_timeout_and_oserror_become_api_errors_status_zero(self):
        c = VeniceClient(api_key="k")
        for error, message in (
            (socket.timeout("slow"), "request timed out"),
            (ConnectionResetError("reset"), "connection error"),
        ):
            with self.subTest(error=type(error).__name__), mock.patch(
                "venice.client.urllib.request.urlopen", side_effect=error
            ), self.assertRaisesRegex(VeniceAPIError, message) as cm:
                c.post_json("/whatever", {})
            self.assertEqual(cm.exception.status, 0)

    def test_urlerror_wrapped_timeout_becomes_api_error_status_zero(self):
        c = VeniceClient(api_key="k")
        with mock.patch(
            "venice.client.urllib.request.urlopen",
            side_effect=URLError(socket.timeout("slow")),
        ), self.assertRaisesRegex(VeniceAPIError, "request timed out") as cm:
            c.post_json("/whatever", {})
        self.assertEqual(cm.exception.status, 0)

    def test_loopback_timeouts_before_headers_and_during_body_are_normalized(self):
        stages = []

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/headers":
                    time.sleep(0.1)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.flush()
                time.sleep(0.1)
                try:
                    self.wfile.write(b'{"ok":true}')
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def log_message(self, format, *args):
                pass

        with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
            thread = threading.Thread(target=server.serve_forever)
            thread.start()
            try:
                client = VeniceClient(
                    api_key="test-fake-key",
                    base_url=f"http://127.0.0.1:{server.server_port}",
                    timeout=0.02,
                )
                for path in ("/headers", "/body"):
                    with self.subTest(path=path), self.assertRaises(
                        VeniceAPIError
                    ) as cm:
                        client.get_json(path)
                    stages.append((path, cm.exception.status, str(cm.exception)))
            finally:
                server.shutdown()
                thread.join()

        self.assertEqual([status for _path, status, _text in stages], [0, 0])
        for _path, _status, text in stages:
            self.assertIn("timed out", text)

    def test_poll_retrieve_returns_audio_after_processing_then_done(self):
        c = VeniceClient(api_key="k")
        sequence = [
            FakeResp(
                200,
                json.dumps(
                    {
                        "status": "PROCESSING",
                        "average_execution_time": 2000,
                        "execution_duration": 500,
                    }
                ).encode(),
                "application/json",
            ),
            FakeResp(200, b"AUDIOBYTES", "audio/mpeg"),
        ]
        ticks = []
        with mock.patch(
            "venice.client.urllib.request.urlopen",
            lambda *a, **kw: sequence.pop(0),
        ), mock.patch("venice.client.time.sleep"):
            ct, audio = c.poll_retrieve(
                "/audio/retrieve",
                {"model": "m", "queue_id": "q"},
                interval=0,
                max_wait=10,
                on_tick=ticks.append,
            )
        self.assertEqual(ct, "audio/mpeg")
        self.assertEqual(audio, b"AUDIOBYTES")
        self.assertEqual(len(ticks), 1)
        self.assertEqual(ticks[0]["status"], "PROCESSING")

    def test_poll_retrieve_treats_unknown_json_status_as_terminal(self):
        c = VeniceClient(api_key="k")
        body = json.dumps({"status": "FAILED", "reason": "infra"}).encode()
        with mock.patch(
            "venice.client.urllib.request.urlopen",
            lambda *a, **kw: FakeResp(200, body, "application/json"),
        ), mock.patch("venice.client.time.sleep"):
            with self.assertRaises(VeniceAPIError):
                c.poll_retrieve(
                    "/audio/retrieve", {"model": "m", "queue_id": "q"},
                    interval=0, max_wait=10,
                )

    def test_request_rejects_non_finite_json_before_urlopen(self):
        c = VeniceClient(api_key="k")
        for bad in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=bad), mock.patch(
                "venice.client.urllib.request.urlopen"
            ) as opened:
                with self.assertRaisesRegex(VeniceAPIError, "strict JSON"):
                    c.post_json("/paid", {"cost": bad})
                opened.assert_not_called()

    def test_poll_retrieve_rejects_non_finite_controls_before_urlopen(self):
        c = VeniceClient(api_key="k")
        for field in ("interval", "max_wait"):
            for bad in (float("nan"), float("inf"), float("-inf")):
                kwargs = {"interval": 0.0, "max_wait": 1.0, field: bad}
                with self.subTest(field=field, value=bad), mock.patch(
                    "venice.client.urllib.request.urlopen"
                ) as opened:
                    with self.assertRaises(ValueError):
                        c.poll_retrieve("/poll", {}, **kwargs)
                    opened.assert_not_called()


if __name__ == "__main__":
    unittest.main()
