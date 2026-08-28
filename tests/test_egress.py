"""Hermetic tests for the pinned public-HTTPS egress boundary (#140)."""
import socket
import unittest
import urllib.error
import urllib.request
from unittest import mock

from venice import _egress


def _answer(address, *, family=socket.AF_INET):
    sockaddr = (address, 443) if family == socket.AF_INET else (address, 443, 0, 0)
    return (family, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", sockaddr)


class TestURLPolicy(unittest.TestCase):
    def test_only_structural_https_without_userinfo_is_accepted(self):
        self.assertEqual(
            _egress.validate_https_url("https://cdn.example/video?sig=x"),
            ("cdn.example", 443),
        )
        for value in (
            "file:///tmp/x",
            "http://example.com/x",
            "data:text/plain,x",
            "https://user:pass@example.com/x",
            "https://example.com:99999/x",
            "https://example.com/x\nheader: value",
            " https://example.com/x",
            "https://example.com/x ",
        ):
            with self.subTest(value=value), self.assertRaises(_egress.EgressPolicyError):
                _egress.validate_https_url(value)

    def test_safe_url_strips_all_bearer_like_components(self):
        rendered = _egress.safe_url(
            "https://user:pass@example.com:8443/video?signature=secret#fragment"
        )
        self.assertEqual(rendered, "https://example.com:8443/video")


class TestResolutionBoundary(unittest.TestCase):
    def test_non_public_ipv4_ipv6_and_mapped_addresses_are_rejected(self):
        addresses = (
            ("127.0.0.1", socket.AF_INET),
            ("10.0.0.1", socket.AF_INET),
            ("169.254.169.254", socket.AF_INET),
            ("224.0.0.1", socket.AF_INET),
            ("0.0.0.0", socket.AF_INET),
            ("192.0.2.1", socket.AF_INET),
            ("::1", socket.AF_INET6),
            ("fe80::1", socket.AF_INET6),
            ("fd00:ec2::254", socket.AF_INET6),
            ("ff02::1", socket.AF_INET6),
            ("::", socket.AF_INET6),
            ("::ffff:127.0.0.1", socket.AF_INET6),
        )
        for address, family in addresses:
            with self.subTest(address=address), mock.patch(
                "venice._egress.socket.getaddrinfo",
                return_value=[_answer(address, family=family)],
            ), self.assertRaises(_egress.EgressPolicyError):
                _egress.resolve_public("cdn.example", 443)

    def test_mixed_public_private_answer_fails_closed(self):
        with mock.patch(
            "venice._egress.socket.getaddrinfo",
            return_value=[_answer("8.8.8.8"), _answer("127.0.0.1")],
        ), self.assertRaises(_egress.EgressPolicyError):
            _egress.resolve_public("rebind.example", 443)

    def test_connection_uses_the_once_resolved_numeric_address(self):
        calls = []

        class FakeSocket:
            def settimeout(self, value):
                calls.append(("timeout", value))

            def bind(self, value):
                calls.append(("bind", value))

            def connect(self, value):
                calls.append(("connect", value))

            def close(self):
                calls.append(("close",))

        resolver = mock.Mock(return_value=[_answer("8.8.8.8")])
        with mock.patch("venice._egress.socket.getaddrinfo", resolver), mock.patch(
            "venice._egress.socket.socket", return_value=FakeSocket()
        ):
            sock = _egress._connect_public("rebind.example", 443, 12)

        self.assertIsInstance(sock, FakeSocket)
        resolver.assert_called_once()
        self.assertIn(("connect", ("8.8.8.8", 443)), calls)

    def test_tls_wrap_keeps_original_hostname_for_verification(self):
        raw = mock.Mock()
        wrapped = mock.Mock()
        connection = _egress.PinnedHTTPSConnection("cdn.example", timeout=5)
        connection._context = mock.Mock()
        connection._context.wrap_socket.return_value = wrapped
        with mock.patch("venice._egress._connect_public", return_value=raw):
            connection.connect()
        connection._context.wrap_socket.assert_called_once_with(
            raw, server_hostname="cdn.example"
        )
        self.assertIs(connection.sock, wrapped)


class TestRedirectAndProxyBoundary(unittest.TestCase):
    def test_redirect_to_cleartext_is_blocked_with_redacted_error(self):
        handler = _egress.GuardedRedirectHandler()
        request = urllib.request.Request("https://public.example/start")
        target = "http://127.0.0.1/private?signature=secret"
        with self.assertRaises(urllib.error.HTTPError) as cm:
            handler.redirect_request(request, None, 302, "Found", {}, target)
        rendered = str(cm.exception)
        self.assertNotIn("signature", rendered)
        self.assertNotIn("secret", rendered)

    def test_redirect_count_and_proxy_policy_are_explicit(self):
        self.assertEqual(_egress.GuardedRedirectHandler.max_redirections, 5)
        opener = _egress.build_https_opener()
        self.assertFalse(
            any(isinstance(h, urllib.request.ProxyHandler) for h in opener.handlers)
        )
        self.assertFalse(
            any(isinstance(h, urllib.request.FileHandler) for h in opener.handlers)
        )
        self.assertFalse(
            any(isinstance(h, urllib.request.HTTPHandler) for h in opener.handlers)
        )
        self.assertTrue(any(isinstance(h, _egress.PinnedHTTPSHandler) for h in opener.handlers))


if __name__ == "__main__":
    unittest.main()
