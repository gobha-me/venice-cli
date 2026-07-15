"""Paths, defaults, env-var names. No I/O, no side effects."""
import os
from pathlib import Path

HOME = Path(os.environ.get("HOME", str(Path.home())))
CONFIG_DIR = HOME / ".config" / "venice"
CREDS_FILE = CONFIG_DIR / "credentials"
PRESETS_FILE = CONFIG_DIR / "image_presets.json"

ENV_API_KEY = "VENICE_API_KEY"
ENV_BASE_URL = "VENICE_BASE_URL"
DEFAULT_BASE_URL = "https://api.venice.ai/api/v1"

SFX_POLL_INTERVAL_SEC = 2.0
SFX_POLL_MAX_WAIT_SEC = 300
