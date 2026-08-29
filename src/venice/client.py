"""Thin Venice.ai HTTP client built on urllib. No third-party deps.

Returns dicts for JSON responses, bytes for binary (audio/image).
Maps non-2xx to VeniceAPIError with status, URL, and a body excerpt.
"""
from __future__ import annotations

import json
import math
import os
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable, Optional, Tuple, Union

from . import __version__, _egress, _numeric, config


VIDEO_DOWNLOAD_MAX_BYTES = 512 * 1024 * 1024
VIDEO_DOWNLOAD_MAX_SECONDS = 15 * 60
VIDEO_DOWNLOAD_IO_TIMEOUT_SECONDS = 60
VIDEO_DOWNLOAD_CHUNK_BYTES = 1024 * 1024
VIDEO_DOWNLOAD_CONTENT_TYPES = frozenset(
    {"video/mp4", "video/webm", "video/quicktime"}
)


class VeniceAPIError(Exception):
    """HTTP-level error from the Venice API.

    Attributes:
        status: HTTP status code (0 if connection failed pre-response).
        url:    request URL with userinfo, query, and fragment redacted.
        body:   excerpt of the response body (first ~2 KB), for debugging.
        code:   Venice API error code (e.g. INSUFFICIENT_BALANCE), if parseable.
    """

    def __init__(self, status: int, url: str, body: str, code: Optional[str] = None):
        self.status = status
        self.url = _egress.safe_url(url) if "://" in str(url) else str(url)
        self.body = body
        self.code = code
        msg = f"HTTP {status} from {self.url}"
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
        user_agent: str = f"venice-cli/{__version__}",
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
            "Accept": "application/json, audio/*, image/*, video/*",
            "User-Agent": self.user_agent,
        }
        data: Optional[bytes] = None
        if json_body is not None:
            try:
                data = json.dumps(json_body, allow_nan=False).encode("utf-8")
            except (TypeError, ValueError) as e:
                raise VeniceAPIError(
                    0, path, f"request body is not valid strict JSON: {e}"
                ) from None
            headers["Content-Type"] = "application/json"

        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        # Bind the credential to this request.  urllib copies regular headers
        # when it constructs a redirected request, but deliberately leaves
        # unredirected headers behind.
        req.add_unredirected_header(
            "Authorization", f"Bearer {self.api_key}"
        )
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
        interval = _numeric.finite_float(interval)
        max_wait = _numeric.finite_float(max_wait)
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

    def download_url_to_temp(
        self,
        url: str,
        directory: Path,
        *,
        max_bytes: int = VIDEO_DOWNLOAD_MAX_BYTES,
        max_seconds: float = VIDEO_DOWNLOAD_MAX_SECONDS,
    ) -> Tuple[str, Path, int]:
        """Safely stream one presigned video URL to a private temporary file.

        The egress opener owns resolution, address policy, pinning, TLS, and redirect
        checks.  This layer owns media type, byte, and wall-time bounds.  It never
        sends the Venice Bearer token and never renders the URL query string.
        """
        try:
            _egress.validate_https_url(url)
        except _egress.EgressPolicyError as e:
            raise VeniceAPIError(0, url, str(e)) from None
        try:
            byte_limit = int(max_bytes)
            time_limit = float(max_seconds)
            configured_timeout = float(self.timeout)
        except (TypeError, ValueError):
            raise VeniceAPIError(0, url, "invalid download resource limit") from None
        if (
            byte_limit <= 0
            or time_limit <= 0
            or not math.isfinite(time_limit)
            or configured_timeout <= 0
            or not math.isfinite(configured_timeout)
        ):
            raise VeniceAPIError(0, url, "invalid download resource limit")
        io_timeout = min(configured_timeout, VIDEO_DOWNLOAD_IO_TIMEOUT_SECONDS)

        req = urllib.request.Request(
            url,
            headers={"Accept": "video/*", "User-Agent": self.user_agent},
            method="GET",
        )
        opener = _egress.build_https_opener()
        tmp_path: Optional[Path] = None
        try:
            with opener.open(req, timeout=io_timeout) as resp:
                ctype = resp.headers.get("Content-Type", "") or ""
                ctype_base = ctype.split(";", 1)[0].strip().lower()
                if ctype_base not in VIDEO_DOWNLOAD_CONTENT_TYPES:
                    raise VeniceAPIError(
                        0, url, f"unsupported video content type: {ctype_base or '(missing)'}"
                    )
                raw_length = resp.headers.get("Content-Length")
                if raw_length:
                    try:
                        declared = int(raw_length)
                    except (TypeError, ValueError):
                        raise VeniceAPIError(0, url, "invalid download Content-Length") from None
                    if declared < 0 or declared > byte_limit:
                        raise VeniceAPIError(
                            0, url, f"video download exceeds {byte_limit} byte limit"
                        )

                directory = Path(directory)
                try:
                    fd, tmp_name = tempfile.mkstemp(
                        prefix=".venice-video-", suffix=".tmp", dir=str(directory)
                    )
                except OSError as e:
                    raise VeniceAPIError(
                        0, url, f"cannot create temporary download file: {e}"
                    ) from None
                tmp_path = Path(tmp_name)
                total = 0
                started = time.monotonic()
                try:
                    with os.fdopen(fd, "wb") as out:
                        while True:
                            if time.monotonic() - started > time_limit:
                                raise VeniceAPIError(0, url, "video download timed out")
                            chunk = resp.read(
                                min(VIDEO_DOWNLOAD_CHUNK_BYTES, byte_limit - total + 1)
                            )
                            if not chunk:
                                break
                            total += len(chunk)
                            if total > byte_limit:
                                raise VeniceAPIError(
                                    0, url, f"video download exceeds {byte_limit} byte limit"
                                )
                            out.write(chunk)
                        if total == 0:
                            raise VeniceAPIError(0, url, "video download was empty")
                        out.flush()
                        os.fsync(out.fileno())
                except Exception:
                    try:
                        tmp_path.unlink()
                    except OSError:
                        pass
                    tmp_path = None
                    raise
                return ctype, tmp_path, total
        except urllib.error.HTTPError as e:
            raise VeniceAPIError(e.code, url, "video download request failed") from None
        except urllib.error.URLError as e:
            if isinstance(e.reason, _egress.EgressPolicyError):
                detail = str(e.reason)
            else:
                detail = "video download connection failed"
            raise VeniceAPIError(0, url, detail) from None
        except _egress.EgressPolicyError as e:
            raise VeniceAPIError(0, url, str(e)) from None
        except (TimeoutError, OSError):
            raise VeniceAPIError(0, url, "video download connection failed") from None

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
