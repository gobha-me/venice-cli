"""Paths, defaults, env-var names. No I/O, no side effects."""
import os
from pathlib import Path

HOME = Path(os.environ.get("HOME", str(Path.home())))
CONFIG_DIR = HOME / ".config" / "venice"
CREDS_FILE = CONFIG_DIR / "credentials"
PRESETS_FILE = CONFIG_DIR / "image_presets.json"
CONFIG_FILE = CONFIG_DIR / "config.json"

ENV_API_KEY = "VENICE_API_KEY"
ENV_BASE_URL = "VENICE_BASE_URL"
DEFAULT_BASE_URL = "https://api.venice.ai/api/v1"

# `venice embed` can target an alternate OpenAI-compatible embeddings endpoint
# (e.g. a local llama.cpp/Ollama/TEI server). These mirror the Venice pair above
# for that opt-in backend; the key is env-only (never config -- local servers
# usually need none, and secrets don't belong in config.json).
ENV_EMBED_BASE_URL = "VENICE_EMBED_BASE_URL"
ENV_EMBED_API_KEY = "VENICE_EMBED_API_KEY"
# Trust a private CA when the alternate backend serves a self-signed cert. Only
# ever applied to --embed-base-url, never the Venice endpoint. (--embed-insecure
# has no env/config knob by design -- disabling verification stays a CLI choice.)
ENV_EMBED_CA_BUNDLE = "VENICE_EMBED_CA_BUNDLE"

# `venice index`/`venice search` keep a semantic-search store *project-local*
# (unlike everything above, which is user-global under ~/.config/venice). The
# store lives at <project>/.venice/index/ -- resolved against a project root at
# runtime so this module stays side-effect-free (no cwd binding at import).
# $VENICE_INDEX_DIR overrides the store location for `search`.
INDEX_DIRNAME = ".venice"
INDEX_SUBDIR = "index"
INDEX_FILENAME = "index.json"
ENV_INDEX_DIR = "VENICE_INDEX_DIR"

# `venice code` (the vcoder coding agent) sandboxes its file/exec tools to a
# project root; $VENICE_CODE_ROOT overrides the default (the current directory).
ENV_CODE_ROOT = "VENICE_CODE_ROOT"

SFX_POLL_INTERVAL_SEC = 2.0
SFX_POLL_MAX_WAIT_SEC = 300

# Video generation runs minutes, not seconds -- poll less often, wait longer.
VIDEO_POLL_INTERVAL_SEC = 5.0
VIDEO_POLL_MAX_WAIT_SEC = 900
