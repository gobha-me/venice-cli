"""Unit tests for VeniceClient. Mocks urllib.request.urlopen."""
import io
import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock
from urllib.error import HTTPError

from venice.client import VeniceAPIError, VeniceClient


class FakeResp:
    def __init__(self, status=200, body=b"", ctype="application/json"):
        self.status = status
        self._body = body
        self.headers = {"Content-Type": ctype}

    def read(self):
        return self._body

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
        self.assertEqual(
            json.loads(captured["body"]),
            {"model": "x", "duration_seconds": 5},
        )

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

    def test_url_error_becomes_venice_api_error_status_zero(self):
        from urllib.error import URLError

        c = VeniceClient(api_key="k")

        def boom(*a, **kw):
            raise URLError("name resolution failed")

        with mock.patch("venice.client.urllib.request.urlopen", boom):
            with self.assertRaises(VeniceAPIError) as cm:
                c.post_json("/whatever", {})
        self.assertEqual(cm.exception.status, 0)

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


if __name__ == "__main__":
    unittest.main()
