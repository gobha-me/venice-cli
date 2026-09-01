"""Hermetic tests for the enforced web/browser egress boundary (#127)."""
import contextlib
import http.server
import os
import socket
import tempfile
import threading
import unittest
from unittest import mock

from venice import _egress
from venice.commands import _browser, _browser_proxy


class _Handler(http.server.BaseHTTPRequestHandler):
    host_headers = []

    def do_GET(self):
        type(self).host_headers.append(self.headers.get("Host"))
        redirects = {
            "/redirect": "http://127.0.0.2/private?token=secret",
            "/redirect-private": "http://10.0.0.1/private?token=secret",
            "/redirect-link-local": "http://169.254.169.254/latest/meta-data/?token=secret",
            "/redirect-metadata": "http://metadata.google.internal/?token=secret",
            "/redirect-mapped": "http://[::ffff:127.0.0.2]/private?token=secret",
        }
        if self.path in redirects:
            self.send_response(302)
            self.send_header("Location", redirects[self.path])
            self.end_headers()
            return
        body = b"<html><script>secret()</script><body>Hello boundary</body></html>"
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, _format, *_args):
        pass


@contextlib.contextmanager
def _origin():
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_address[1]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _private_policy(**kwargs):
    values = {
        "private_hosts": ["127.0.0.1"],
        "private_ranges": ["127.0.0.1/32"],
    }
    values.update(kwargs)
    return _egress.DestinationPolicy.create(**values)


class TestDestinationPolicy(unittest.TestCase):
    def test_private_access_requires_exact_host_and_range(self):
        for hosts, ranges in (([], []), (["127.0.0.1"], []), ([], ["127.0.0.1/32"])):
            policy = _egress.DestinationPolicy.create(
                private_hosts=hosts, private_ranges=ranges
            )
            with self.subTest(hosts=hosts, ranges=ranges), self.assertRaises(
                _egress.EgressPolicyError
            ):
                policy.resolve("127.0.0.1", 80)
        self.assertEqual(_private_policy().resolve("127.0.0.1", 80)[0][3], ("127.0.0.1", 80))

    def test_private_range_authority_is_narrow(self):
        for value in ("0.0.0.0/0", "8.8.8.0/24", "169.254.0.0/16", "ff00::/8"):
            with self.subTest(value=value), self.assertRaises(_egress.EgressPolicyError):
                _egress.DestinationPolicy.create(private_ranges=[value])

    def test_deny_wins_and_metadata_names_are_hard_blocked(self):
        policy = _egress.DestinationPolicy.create(
            allow=["*.example"], deny=["secret.example"],
            private_hosts=["metadata.google.internal"], private_ranges=["10.0.0.0/8"],
        )
        with self.assertRaises(_egress.EgressPolicyError):
            policy.endpoint("https://secret.example/")
        with self.assertRaises(_egress.EgressPolicyError):
            policy.endpoint("http://metadata.google.internal/")

    def test_mixed_dns_answer_fails_closed(self):
        answer = lambda value: (socket.AF_INET, socket.SOCK_STREAM, 6, "", (value, 443))
        policy = _egress.DestinationPolicy.create()
        with mock.patch("venice._egress.socket.getaddrinfo", return_value=[
            answer("8.8.8.8"), answer("127.0.0.1")
        ]), self.assertRaises(_egress.EgressPolicyError):
            policy.resolve("rebind.example", 443)

    def test_mapped_loopback_and_metadata_addresses_are_never_authorized(self):
        policy = _egress.DestinationPolicy.create(
            private_hosts=["::ffff:127.0.0.1", "169.254.169.254", "100.100.100.200"],
            private_ranges=["127.0.0.0/8", "10.0.0.0/8"],
        )
        for host in ("::ffff:127.0.0.1", "169.254.169.254", "100.100.100.200"):
            with self.subTest(host=host), self.assertRaises(_egress.EgressPolicyError):
                policy.resolve(host, 80)


class TestWebFetch(unittest.TestCase):
    def test_authorized_loopback_fetch_and_size_cap(self):
        _Handler.host_headers = []
        with _origin() as port:
            result = _browser.web_fetch(
                f"http://127.0.0.1:{port}/", max_bytes=10_000,
                private_hosts=["127.0.0.1"], private_ranges=["127.0.0.1/32"],
            )
        self.assertTrue(result["ok"], result)
        self.assertEqual(result["text"], "Hello boundary")
        self.assertFalse(result["truncated"])
        self.assertEqual(_Handler.host_headers, [f"127.0.0.1:{port}"])

    def test_redirects_to_special_destinations_are_blocked_without_contact(self):
        with _origin() as port:
            for path in (
                "/redirect", "/redirect-private", "/redirect-link-local",
                "/redirect-metadata", "/redirect-mapped",
            ):
                with self.subTest(path=path):
                    result = _browser.web_fetch(
                        f"http://127.0.0.1:{port}{path}",
                        private_hosts=["127.0.0.1"],
                        private_ranges=["127.0.0.1/32"],
                    )
                    self.assertFalse(result["ok"])
                    self.assertNotIn("secret", result["error"])

    def test_no_ambient_proxy_or_alternate_scheme_handlers(self):
        opener = _egress.build_policy_opener(_egress.DestinationPolicy.create())
        names = {type(handler).__name__ for handler in opener.handlers}
        self.assertNotIn("ProxyHandler", names)
        self.assertNotIn("FileHandler", names)
        self.assertNotIn("FTPHandler", names)


class TestPolicyProxy(unittest.TestCase):
    def test_absolute_http_request_reaches_only_authorized_origin(self):
        with _origin() as port, _browser_proxy.policy_proxy(_private_policy()) as proxy:
            proxy_port = int(proxy.rsplit(":", 1)[1])
            with socket.create_connection(("127.0.0.1", proxy_port), timeout=3) as sock:
                sock.sendall(
                    f"GET http://127.0.0.1:{port}/ HTTP/1.1\r\n"
                    f"Host: 127.0.0.1:{port}\r\nConnection: close\r\n\r\n".encode()
                )
                response = b""
                while True:
                    chunk = sock.recv(8192)
                    if not chunk:
                        break
                    response += chunk
        self.assertIn(b"200 OK", response)
        self.assertIn(b"Hello boundary", response)

    def test_connect_to_special_destinations_is_forbidden(self):
        with _browser_proxy.policy_proxy(_private_policy()) as proxy:
            proxy_port = int(proxy.rsplit(":", 1)[1])
            for authority in (
                "127.0.0.2:443", "10.0.0.1:443", "169.254.169.254:443",
                "100.100.100.200:443", "[::ffff:127.0.0.2]:443",
            ):
                with self.subTest(authority=authority), socket.create_connection(
                    ("127.0.0.1", proxy_port), timeout=3
                ) as sock:
                    sock.sendall(
                        f"CONNECT {authority} HTTP/1.1\r\nHost: {authority}\r\n\r\n".encode()
                    )
                    response = sock.recv(1024)
                    self.assertIn(b"403 Forbidden", response)


class TestChromiumContainment(unittest.TestCase):
    def test_argv_has_proxy_and_sandbox_without_no_sandbox(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            _browser, "find_browser", return_value=("/usr/bin/chromium", "chromium")
        ), mock.patch.object(
            _browser._egress.DestinationPolicy, "resolve", return_value=[]
        ), mock.patch.object(
            _browser._browser_proxy, "policy_proxy", return_value=contextlib.nullcontext("http://127.0.0.1:9999")
        ), mock.patch.object(
            _browser, "_run", return_value=(0, "<html>ok</html>", "", None)
        ) as run:
            result = _browser.capture("https://example.com/", mode="dom", timeout=3)
        self.assertTrue(result["ok"], result)
        argv = run.call_args.args[0]
        self.assertIn("--proxy-server=http://127.0.0.1:9999", argv)
        self.assertIn("--proxy-bypass-list=<-loopback>", argv)
        self.assertIn("--disable-quic", argv)
        self.assertIn("--force-webrtc-ip-handling-policy=disable_non_proxied_udp", argv)
        self.assertNotIn("--no-sandbox", argv)

    def test_browser_environment_drops_credentials_and_proxies(self):
        with mock.patch.dict(os.environ, {
            "VENICE_API_KEY": "do-not-copy", "HTTPS_PROXY": "http://proxy",
            "PATH": "/bin",
        }, clear=True):
            env = _browser._browser_env("/tmp/disposable-profile")
        self.assertEqual(env["PATH"], "/bin")
        self.assertEqual(env["HOME"], "/tmp/disposable-profile")
        self.assertNotIn("VENICE_API_KEY", env)
        self.assertNotIn("HTTPS_PROXY", env)


if __name__ == "__main__":
    unittest.main()
