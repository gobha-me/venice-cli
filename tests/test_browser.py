"""Unit tests for the web-fetch + headless-browser rails (`_browser`, #71).

Hermetic: the URL policy and html->text are pure; web_fetch mocks the urllib opener and
capture mocks `subprocess.run` + `find_browser`, so no network, no real browser, no real
key. stdlib-only, runs on the 3.9 floor.
"""
import os
import subprocess
import tempfile
import unittest
import urllib.error
from unittest import mock

from venice.commands import _browser


class _FakeResp:
    """A urllib response stand-in supporting the context-manager + read/geturl/headers API."""

    def __init__(self, body=b"", *, url="http://example.com/", ctype="text/html; charset=utf-8"):
        self._body = body
        self._url = url
        self.headers = {"Content-Type": ctype}

    def read(self, n=-1):
        return self._body if (n is None or n < 0) else self._body[:n]

    def geturl(self):
        return self._url

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


# --------------------------------------------------------------------------- #
# URL policy
# --------------------------------------------------------------------------- #
class TestUrlPolicy(unittest.TestCase):
    def test_http_https_allowed(self):
        self.assertIsNone(_browser.check_url_policy("http://localhost:8123/"))
        self.assertIsNone(_browser.check_url_policy("https://example.com/x"))

    def test_nonhttp_scheme_blocked(self):
        for u in ("file:///etc/passwd", "ftp://h/x", "data:text/html,x", "gopher://h"):
            with self.subTest(u=u):
                msg = _browser.check_url_policy(u)
                self.assertIsNotNone(msg)
                self.assertIn("scheme", msg)

    def test_metadata_endpoint_blocked(self):
        for u in ("http://169.254.169.254/latest/meta-data/",
                  "http://metadata.google.internal/x"):
            with self.subTest(u=u):
                self.assertIsNotNone(_browser.check_url_policy(u))

    def test_metadata_encoding_bypasses_blocked(self):
        # SSRF hardening: integer/hex IPv4 encodings of the link-local range and the AWS
        # IPv6 metadata literal must be refused, not just the dotted string.
        for u in ("http://2852039166/latest/",          # decimal 169.254.169.254
                  "http://0xA9FEA9FE/",                  # hex 169.254.169.254
                  "http://169.254.0.1/",                 # elsewhere in link-local
                  "http://[fd00:ec2::254]/latest/"):     # AWS IPv6 metadata
            with self.subTest(u=u):
                self.assertIsNotNone(_browser.check_url_policy(u))

    def test_normal_hosts_and_localhost_not_falsely_blocked(self):
        for u in ("http://localhost:8123/", "http://127.0.0.1/", "https://example.com/"):
            with self.subTest(u=u):
                self.assertIsNone(_browser.check_url_policy(u))

    def test_empty_and_hostless(self):
        self.assertIn("required", _browser.check_url_policy(""))
        self.assertIsNotNone(_browser.check_url_policy("http:///nohost"))

    def test_deny_glob_wins_and_matches_host_or_url(self):
        self.assertIn("deny", _browser.check_url_policy("http://evil.com/x", deny=["evil.com"]))
        self.assertIn("deny", _browser.check_url_policy("http://a/secret", deny=["*secret*"]))
        # deny wins even when the host is also allow-listed
        self.assertIsNotNone(
            _browser.check_url_policy("http://evil.com", allow=["evil.com"], deny=["evil.com"]))

    def test_allowlist_restricts_hosts(self):
        self.assertIsNone(_browser.check_url_policy("http://example.com/x", allow=["example.com"]))
        self.assertIsNone(_browser.check_url_policy("http://api.example.com", allow=["*.example.com"]))
        self.assertIn("not in the browser allowlist",
                      _browser.check_url_policy("http://other.com", allow=["example.com"]))


# --------------------------------------------------------------------------- #
# Binary probe + env + helpers
# --------------------------------------------------------------------------- #
class TestProbeAndEnv(unittest.TestCase):
    def test_prefers_chromium_over_firefox(self):
        present = {"chromium", "firefox"}
        with mock.patch("venice.commands._browser.shutil.which",
                        side_effect=lambda n: "/usr/bin/" + n if n in present else None):
            self.assertEqual(_browser.find_browser(), ("/usr/bin/chromium", "chromium"))

    def test_firefox_is_last_resort(self):
        with mock.patch("venice.commands._browser.shutil.which",
                        side_effect=lambda n: "/usr/bin/firefox" if n == "firefox" else None):
            self.assertEqual(_browser.find_browser(), ("/usr/bin/firefox", "firefox"))

    def test_none_when_absent(self):
        with mock.patch("venice.commands._browser.shutil.which", return_value=None):
            self.assertIsNone(_browser.find_browser())

    def test_env_allowlist_drops_secret_and_ambient(self):
        fake = {"VENICE_API_KEY": "test-fake-key", "GH_TOKEN": "x", "PATH": "/usr/bin",
                "HOME": "/home/u", "LC_CTYPE": "C.UTF-8"}
        with mock.patch.dict(os.environ, fake, clear=True):
            env = _browser._browser_env()
        self.assertNotIn("VENICE_API_KEY", env)   # the credential never reaches the child
        self.assertNotIn("GH_TOKEN", env)          # nor any other ambient token
        self.assertEqual(env.get("PATH"), "/usr/bin")
        self.assertEqual(env.get("HOME"), "/home/u")
        self.assertEqual(env.get("LC_CTYPE"), "C.UTF-8")   # LC_* passed by prefix

    def test_html_to_text(self):
        t = _browser.html_to_text(
            "<p>Hello &amp; <b>world</b></p><script>bad()</script><style>x{}</style>")
        self.assertIn("Hello & world", t)
        self.assertNotIn("bad()", t)
        self.assertNotIn("x{}", t)

    def test_capture_filename(self):
        self.assertEqual(_browser.capture_filename("http://localhost:8123/a/b?c=1"),
                         "capture-localhost.png")
        self.assertTrue(_browser.capture_filename("not a url").endswith(".png"))


# --------------------------------------------------------------------------- #
# web_fetch
# --------------------------------------------------------------------------- #
class TestWebFetch(unittest.TestCase):
    def _opener(self, resp):
        op = mock.Mock()
        op.open.return_value = resp
        return op

    def test_text_mode_strips_tags(self):
        body = b"<html><body><h1>Title</h1><p>Body text</p></body></html>"
        op = self._opener(_FakeResp(body))
        with mock.patch("venice.commands._browser.urllib.request.build_opener", return_value=op):
            res = _browser.web_fetch("http://example.com/", mode="text")
        self.assertTrue(res["ok"])
        self.assertIn("Title", res["text"])
        self.assertIn("Body text", res["text"])
        self.assertNotIn("<h1>", res["text"])

    def test_html_mode_keeps_markup(self):
        op = self._opener(_FakeResp(b"<h1>Raw</h1>"))
        with mock.patch("venice.commands._browser.urllib.request.build_opener", return_value=op):
            res = _browser.web_fetch("http://example.com/", mode="html")
        self.assertIn("<h1>Raw</h1>", res["html"])

    def test_blocked_url_never_opens(self):
        op = self._opener(_FakeResp(b"x"))
        with mock.patch("venice.commands._browser.urllib.request.build_opener", return_value=op):
            res = _browser.web_fetch("file:///etc/passwd")
        self.assertFalse(res["ok"])
        op.open.assert_not_called()

    def test_http_error_returned_not_raised(self):
        op = mock.Mock()
        op.open.side_effect = urllib.error.HTTPError("http://x/", 404, "Not Found", {}, None)
        with mock.patch("venice.commands._browser.urllib.request.build_opener", return_value=op):
            res = _browser.web_fetch("http://x/")
        self.assertFalse(res["ok"])
        self.assertIn("404", res["error"])

    def test_redirect_guard_blocks_metadata_hop(self):
        # urllib auto-follows redirects; a 302 -> the metadata endpoint must be refused.
        h = _browser._GuardedRedirect(allow=(), deny=())
        with self.assertRaises(urllib.error.HTTPError):
            h.redirect_request(None, None, 302, "Found", {}, "http://169.254.169.254/latest/")


# --------------------------------------------------------------------------- #
# capture (subprocess mocked)
# --------------------------------------------------------------------------- #
class TestCapture(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.mkdtemp()
        self.addCleanup(__import__("shutil").rmtree, self.td, ignore_errors=True)

    def test_dom_chromium_argv_env_and_assert(self):
        seen = {}

        def run(argv, **kw):
            seen["argv"], seen["env"] = argv, kw.get("env")
            return mock.Mock(returncode=0, stdout="<html>DEAD DISTRICT</html>", stderr="")

        with mock.patch.object(_browser, "find_browser",
                               return_value=("/usr/bin/chromium", "chromium")), \
                mock.patch.dict(os.environ, {"VENICE_API_KEY": "test-fake-key"}), \
                mock.patch("venice.commands._browser.subprocess.run", side_effect=run):
            res = _browser.capture("http://localhost:8123/", mode="dom",
                                   assert_contains="DEAD DISTRICT")
        self.assertTrue(res["ok"])
        self.assertEqual(res["family"], "chromium")
        self.assertTrue(res["contains"])
        self.assertIn("DEAD DISTRICT", res["dom"])
        self.assertIn("--dump-dom", seen["argv"])
        self.assertIn("--headless=new", seen["argv"])
        self.assertTrue(any(a.startswith("--user-data-dir=") for a in seen["argv"]))
        self.assertNotIn("VENICE_API_KEY", seen["env"])   # scrubbed env into the child

    def test_assert_contains_uses_full_dom_before_truncation(self):
        needle = "MARKER_PAST_CAP"
        big = "<html>" + ("x" * (_browser.MAX_OUTPUT_CHARS + 50)) + needle + "</html>"
        with mock.patch.object(_browser, "find_browser",
                               return_value=("/usr/bin/chromium", "chromium")), \
                mock.patch("venice.commands._browser.subprocess.run",
                           return_value=mock.Mock(returncode=0, stdout=big, stderr="")):
            res = _browser.capture("http://localhost/", mode="dom", assert_contains=needle)
        self.assertTrue(res["contains"])            # matched despite the returned dom being capped
        self.assertTrue(res["truncated"])
        self.assertLessEqual(len(res["dom"]), _browser.MAX_OUTPUT_CHARS)

    def test_screenshot_firefox_argv_and_writes_file(self):
        out = os.path.join(self.td, "shot.png")
        seen = {}

        def run(argv, **kw):
            seen["argv"] = argv
            with open(out, "wb") as f:
                f.write(b"\x89PNG\r\n")
            return mock.Mock(returncode=0, stdout="", stderr="")

        with mock.patch.object(_browser, "find_browser",
                               return_value=("/usr/bin/firefox", "firefox")), \
                mock.patch("venice.commands._browser.subprocess.run", side_effect=run):
            res = _browser.capture("http://localhost/", out_path=out, mode="screenshot")
        self.assertTrue(res["ok"])
        self.assertEqual(res["screenshot_path"], out)
        self.assertIn("--screenshot", seen["argv"])
        self.assertNotIn("--dump-dom", seen["argv"])

    def test_dom_on_firefox_rejected(self):
        with mock.patch.object(_browser, "find_browser",
                               return_value=("/usr/bin/firefox", "firefox")):
            res = _browser.capture("http://localhost/", mode="dom")
        self.assertFalse(res["ok"])
        self.assertIn("screenshot-only", res["error"])

    def test_no_browser_degrades(self):
        with mock.patch.object(_browser, "find_browser", return_value=None):
            res = _browser.capture("http://localhost/", mode="dom")
        self.assertFalse(res["ok"])
        self.assertIn("no headless browser", res["error"])

    def test_binary_vanished_degrades(self):
        with mock.patch.object(_browser, "find_browser",
                               return_value=("/usr/bin/chromium", "chromium")), \
                mock.patch("venice.commands._browser.subprocess.run",
                           side_effect=FileNotFoundError()):
            res = _browser.capture("http://localhost/", mode="dom")
        self.assertFalse(res["ok"])
        self.assertIn("no headless browser", res["error"])

    def test_timeout(self):
        with mock.patch.object(_browser, "find_browser",
                               return_value=("/usr/bin/chromium", "chromium")), \
                mock.patch("venice.commands._browser.subprocess.run",
                           side_effect=subprocess.TimeoutExpired("x", 1)):
            res = _browser.capture("http://localhost/", mode="dom")
        self.assertFalse(res["ok"])
        self.assertIn("timed out", res["error"])

    def test_blocked_url_checked_before_probe(self):
        with mock.patch.object(_browser, "find_browser") as fb:
            res = _browser.capture("file:///etc/passwd", mode="dom")
        self.assertFalse(res["ok"])
        fb.assert_not_called()

    def test_unknown_mode(self):
        res = _browser.capture("http://localhost/", mode="bogus")
        self.assertFalse(res["ok"])
        self.assertIn("unknown mode", res["error"])


if __name__ == "__main__":
    unittest.main()
