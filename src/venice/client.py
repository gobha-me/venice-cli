"""Thin Venice.ai HTTP client built on urllib. No third-party deps.

Returns dicts for JSON responses, bytes for binary (audio/image).
Maps non-2xx to VeniceAPIError with status, URL, and a body excerpt.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Callable, Optional, Tuple, Union

from . import config


class VeniceAPIError(Exception):
    """HTTP-level error from the Venice API.

    Attributes:
        status: HTTP status code (0 if connection failed pre-response).
        url:    final request URL.
        body:   excerpt of the response body (first ~2 KB), for debugging.
        code:   Venice API error code (e.g. INSUFFICIENT_BALANCE), if parseable.
    """

    def __init__(self, status: int, url: str, body: str, code: Optional[str] = None):
        self.status = status
        self.url = url
        self.body = body
        self.code = code
        msg = f"HTTP {status} from {url}"
        if code:
            msg += f" [{code}]"
        if body:
            msg += f"\n  body: {body[:500]}"
        super().__init__(msg)


ResponseType = Union[dict, bytes]


class VeniceClient:
    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        timeout: float = 60.0,
        user_agent: str = "venice-cli/0.1",
    ):
        if not api_key:
            raise ValueError("api_key is required")
        self.api_key = api_key
        self.base_url = (base_url or config.DEFAULT_BASE_URL).rstrip("/")
        self.timeout = timeout
        self.user_agent = user_agent

    def request(
        self,
        method: str,
        path: str,
        *,
        json_body: Optional[dict] = None,
        params: Optional[dict] = None,
    ) -> Tuple[int, str, bytes]:
        url = self.base_url + path
        if params:
            url += "?" + urllib.parse.urlencode(params, doseq=True)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json, audio/*, image/*, video/*",
            "User-Agent": self.user_agent,
        }
        data: Optional[bytes] = None
        if json_body is not None:
            data = json.dumps(json_body).encode("utf-8")
            headers["Content-Type"] = "application/json"

        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = resp.read()
                ctype = resp.headers.get("Content-Type", "")
                status = getattr(resp, "status", 200)
                return status, ctype, body
        except urllib.error.HTTPError as e:
            err_body = b""
            try:
                err_body = e.read()
            except Exception:
                pass
            err_ctype = ""
            try:
                err_ctype = e.headers.get("Content-Type", "")
            except Exception:
                pass
            self._raise_api_error(e.code, url, err_body, err_ctype)
        except urllib.error.URLError as e:
            raise VeniceAPIError(0, url, f"connection error: {e.reason}") from None

    def post_json(self, path: str, body: dict) -> dict:
        status, ctype, raw = self.request("POST", path, json_body=body)
        return self._decode_json(status, path, ctype, raw)

    def get_json(self, path: str, params: Optional[dict] = None) -> dict:
        status, ctype, raw = self.request("GET", path, params=params)
        return self._decode_json(status, path, ctype, raw)

    def post_for_bytes_or_json(
        self, path: str, body: dict
    ) -> Tuple[str, ResponseType]:
        """For endpoints that may return JSON (in-progress) OR binary (done).

        Used by /audio/retrieve. Returns (content_type, payload):
          - ("audio/mpeg", b"...") on completion
          - ("application/json", {...}) while still processing
        """
        status, ctype, raw = self.request("POST", path, json_body=body)
        ct_low = (ctype or "").lower()
        if ct_low.startswith("application/json"):
            return ctype, (json.loads(raw.decode("utf-8")) if raw else {})
        if (
            ct_low.startswith("audio/")
            or ct_low.startswith("image/")
            or ct_low.startswith("video/")
        ):
            return ctype, raw
        return ctype, raw

    def poll_retrieve(
        self,
        path: str,
        body: dict,
        *,
        interval: float = config.SFX_POLL_INTERVAL_SEC,
        max_wait: float = config.SFX_POLL_MAX_WAIT_SEC,
        on_tick: Optional[Callable[[dict], None]] = None,
        terminal_statuses: Tuple[str, ...] = (),
    ) -> Tuple[str, ResponseType]:
        """Poll an async endpoint that switches content-type on completion.

        On success returns (content_type, payload):
          - (ctype, bytes) when the endpoint streams the finished media, or
          - (ctype, dict) when the JSON `status` is in `terminal_statuses`
            (e.g. video's "COMPLETED" -- the media is fetched separately from a
            download_url). Audio callers leave `terminal_statuses` empty and
            always get bytes.

        Raises VeniceAPIError on terminal HTTP errors or an unexpected status.
        Raises TimeoutError if max_wait elapses while still PROCESSING.
        """
        deadline = time.monotonic() + max_wait
        while True:
            ctype, payload = self.post_for_bytes_or_json(path, body)
            if isinstance(payload, (bytes, bytearray)):
                return ctype, bytes(payload)
            if not isinstance(payload, dict):
                raise VeniceAPIError(
                    0, path, f"unexpected payload type from {path}: {type(payload).__name__}"
                )
            status = payload.get("status")
            if status and status in terminal_statuses:
                return ctype, payload
            if status and status != "PROCESSING":
                raise VeniceAPIError(
                    0, path, f"unexpected status: {payload!r}"
                )
            if on_tick:
                try:
                    on_tick(payload)
                except Exception:
                    pass
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"not ready after {max_wait}s "
                    f"(last status: {status!r})"
                )
            time.sleep(interval)

    def get_url_bytes(self, url: str) -> Tuple[str, bytes]:
        """Fetch an arbitrary URL's bytes with a plain GET (no auth header).

        For presigned download URLs (e.g. a video's download_url), which carry
        their own auth in the query string and must not receive our Bearer token.
        Returns (content_type, bytes); maps HTTP/URL errors to VeniceAPIError.
        """
        req = urllib.request.Request(
            url, headers={"User-Agent": self.user_agent}, method="GET"
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return resp.headers.get("Content-Type", ""), resp.read()
        except urllib.error.HTTPError as e:
            err_body = b""
            try:
                err_body = e.read()
            except Exception:
                pass
            self._raise_api_error(e.code, url, err_body, "")
        except urllib.error.URLError as e:
            raise VeniceAPIError(0, url, f"connection error: {e.reason}") from None

    @staticmethod
    def _decode_json(status: int, path: str, ctype: str, raw: bytes) -> dict:
        if not raw:
            return {}
        try:
            return json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            raise VeniceAPIError(
                status, path, f"non-JSON response ({ctype}): {e}"
            ) from None

    def get_balance(self) -> Optional[dict]:
        """Fetch current balance + tier via /api_keys/rate_limits.

        Returns the parsed `data` block (with balances, apiTier, nextEpochBegins,
        rateLimits) or None if the call fails. Best-effort; callers should
        treat None as "balance unavailable, continue".
        """
        try:
            doc = self.get_json("/api_keys/rate_limits")
        except VeniceAPIError:
            return None
        data = doc.get("data") if isinstance(doc, dict) else None
        return data if isinstance(data, dict) else None

    @staticmethod
    def _raise_api_error(status: int, url: str, body: bytes, ctype: str):
        excerpt = ""
        code: Optional[str] = None
        try:
            text = body.decode("utf-8", errors="replace")
            excerpt = text[:2048]
            if (ctype or "").lower().startswith("application/json"):
                doc: Any = json.loads(text)
                if isinstance(doc, dict):
                    code = doc.get("code")
                    if not code and isinstance(doc.get("error"), dict):
                        code = doc["error"].get("code")
        except Exception:
            pass
        raise VeniceAPIError(status, url, excerpt, code=code)


def build_client_from_auth():
    """Construct a VeniceClient using env-var or file credentials.

    Raises auth.AuthError if no key is available. Honors $VENICE_BASE_URL.
    """
    from . import auth

    key = auth.load_key()
    base = os.environ.get(config.ENV_BASE_URL) or config.DEFAULT_BASE_URL
    return VeniceClient(api_key=key, base_url=base)
