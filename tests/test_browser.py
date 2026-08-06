"""Containment tests for the security-disabled browser rail.

Hermetic: these regressions prove both public entry points fail closed without DNS,
HTTP, or a browser subprocess while GHSA-mqjr-2vh8-6fvg is contained.
"""
import socket
import subprocess
import unittest
import urllib.request
from unittest import mock

from venice.commands import _browser


class TestBrowserContainment(unittest.TestCase):
    def _assert_disabled(self, result):
        self.assertFalse(result["ok"])
        self.assertIn("temporarily disabled for security", result["error"])
        self.assertIn("GHSA-mqjr-2vh8-6fvg", result["error"])

    def test_web_fetch_always_fails_closed(self):
        for url in (
            "https://example.com/",
            "http://127.0.0.1/",
            "http://169.254.169.254/latest/meta-data/",
            "file:///etc/passwd",
        ):
            with self.subTest(url=url):
                self._assert_disabled(_browser.web_fetch(url))

    def test_capture_always_fails_closed(self):
        for mode in ("dom", "text", "screenshot", "both"):
            with self.subTest(mode=mode):
                self._assert_disabled(_browser.capture(
                    "https://example.com/", mode=mode, out_path="unused.png"))

    def test_entry_points_never_touch_network_or_processes(self):
        with mock.patch.object(urllib.request, "urlopen") as urlopen, \
                mock.patch.object(urllib.request, "build_opener") as build_opener, \
                mock.patch.object(socket, "getaddrinfo") as getaddrinfo, \
                mock.patch.object(subprocess, "run") as run:
            self._assert_disabled(_browser.web_fetch("https://example.com/"))
            self._assert_disabled(_browser.capture("https://example.com/"))
        urlopen.assert_not_called()
        build_opener.assert_not_called()
        getaddrinfo.assert_not_called()
        run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
