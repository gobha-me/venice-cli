"""Real-Chromium egress containment test; required by the dedicated CI job."""
import http.server
import os
import socket
import threading
import unittest
from unittest import mock

from venice.commands import _browser


class _Allowed(http.server.BaseHTTPRequestHandler):
    blocked_port = 0

    def do_GET(self):
        if self.path == "/ok":
            body = b"BOUNDARY_OK"
        elif self.path == "/redirect":
            self.send_response(302)
            self.send_header(
                "Location", f"http://127.0.0.2:{self.blocked_port}/redirected"
            )
            self.end_headers()
            return
        else:
            target = f"http://127.0.0.2:{self.blocked_port}"
            alternate = str(127 * 256 ** 3 + 2)
            body = f"""<!doctype html><html><head>
<link rel=stylesheet href='{target}/style.css'>
<link rel=prefetch href='{target}/prefetch'>
<script src='{target}/script.js'></script></head><body>
<img src='{target}/image.png'><img src='http://{alternate}:{self.blocked_port}/alternate.png'>
<iframe src='{target}/frame'></iframe><iframe src='/redirect'></iframe>
<a id=d download href='{target}/download'>download</a><div id=result>waiting</div>
<script>
fetch('/ok').then(r => r.text()).then(t => document.querySelector('#result').textContent=t);
fetch('{target}/fetch').catch(()=>{{}});
fetch('http://[::ffff:127.0.0.2]/mapped').catch(()=>{{}});
try {{ let x=new XMLHttpRequest(); x.open('GET','{target}/xhr'); x.send(); }} catch(e) {{}}
try {{ new Worker('{target}/worker.js'); }} catch(e) {{}}
try {{ navigator.serviceWorker.register('{target}/sw.js'); }} catch(e) {{}}
try {{ new WebSocket('ws://127.0.0.2:{self.blocked_port}/socket'); }} catch(e) {{}}
document.querySelector('#d').click();
</script></body></html>""".encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, _format, *_args):
        pass


class _Blocked(http.server.BaseHTTPRequestHandler):
    count = 0
    lock = threading.Lock()

    def _hit(self):
        with self.lock:
            type(self).count += 1
        self.send_response(204)
        self.end_headers()

    do_GET = _hit

    def log_message(self, _format, *_args):
        pass


class TestRealBrowserBoundary(unittest.TestCase):
    def test_all_page_traffic_is_confined_to_policy_proxy(self):
        required = os.environ.get("VENICE_BROWSER_REQUIRED") == "1"
        if not required:
            self.skipTest("real browser containment runs in its dedicated CI job")
        if not _browser.find_browser():
            self.fail("dedicated browser job requires a Chromium-family browser")
        blocked = http.server.ThreadingHTTPServer(("127.0.0.2", 0), _Blocked)
        _Allowed.blocked_port = blocked.server_address[1]
        allowed = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Allowed)
        _Blocked.count = 0
        threads = [
            threading.Thread(target=server.serve_forever, daemon=True)
            for server in (allowed, blocked)
        ]
        for thread in threads:
            thread.start()
        try:
            def resolve(host, port, **_kwargs):
                if host != "fixture.test":
                    raise AssertionError(f"unexpected DNS lookup outside proxy policy: {host}")
                return [(
                    socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "",
                    ("127.0.0.1", port),
                )]

            with mock.patch("venice._egress.socket.getaddrinfo", side_effect=resolve):
                result = _browser.capture(
                    f"http://fixture.test:{allowed.server_address[1]}/",
                    mode="dom", wait_ms=3000, timeout=45,
                    assert_contains="BOUNDARY_OK",
                    allow=["fixture.test"],
                    private_hosts=["fixture.test"],
                    private_ranges=["127.0.0.1/32"],
                )
        finally:
            for server in (allowed, blocked):
                server.shutdown()
                server.server_close()
            for thread in threads:
                thread.join(timeout=5)
        self.assertTrue(result.get("ok"), result)
        self.assertTrue(result.get("contains"), result)
        self.assertEqual(_Blocked.count, 0, "Chromium reached an unauthorized loopback peer")


if __name__ == "__main__":
    unittest.main()
