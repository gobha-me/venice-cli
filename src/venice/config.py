"""Paths, defaults, env-var names. No I/O, no side effects."""
import os
from pathlib import Path

HOME = Path(os.environ.get("HOME", str(Path.home())))
CONFIG_DIR = HOME / ".config" / "venice"
CREDS_FILE = CONFIG_DIR / "credentials"
PRESETS_FILE = CONFIG_DIR / "image_presets.json"
CONFIG_FILE = CONFIG_DIR / "config.json"
# Structured 0600 store for NAMED secrets beyond the single Venice key in
# `credentials` (#43): the embed-backend key, and later MCP/cluster tokens. Kept
# separate from config.json (plaintext prefs) so secrets never land in the config.
SECRETS_FILE = CONFIG_DIR / "secrets.json"
# `venice chat` reads local, file-backed system prompts ("personas", #68) from a
# dedicated subdir so listing them never has to enumerate the config root (where
# `credentials` lives). Names resolve to <PERSONAS_DIR>/<name>.md|.txt only.
PERSONAS_DIR = CONFIG_DIR / "personas"

# `venice chat`/`venice code` auto-save each REPL session (id + settings + usage +
# transcript) here (#47), so `--resume <id>` / `--continue` restore a session, not
# just its messages. One JSON envelope per session at <SESSIONS_DIR>/<id>.json,
# written 0600 (mirrors the credential store's hygiene, though transcripts are not
# secrets). $VENICE_SESSIONS_DIR overrides the location (mirrors $VENICE_INDEX_DIR),
# resolved at runtime so this module stays side-effect-free.
SESSIONS_DIR = CONFIG_DIR / "sessions"
ENV_SESSIONS_DIR = "VENICE_SESSIONS_DIR"

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

# Persistent agent memory + task list (#49): the chat/code agent's own durable
# notes and checklist, so multi-step and cross-session work (and the #52 planner
# handing state to subagents) survives beyond one transcript. TWO TIERS:
#   - GLOBAL memory (knowledge that travels with the agent) lives user-global at
#     <MEMORY_DIR>/memory.json; $VENICE_MEMORY_DIR overrides (mirrors the sessions
#     pair), resolved at runtime so this module stays side-effect-free.
#   - PROJECT memory + the task list ride the repo at <project>/.venice/memory/
#     (discovered like the .venice index), so subagents in the same tree share them.
# Both files are name-keyed JSON maps written 0600 (mirrors the store hygiene above).
MEMORY_DIR = CONFIG_DIR / "memory"
ENV_MEMORY_DIR = "VENICE_MEMORY_DIR"
MEMORY_SUBDIR = "memory"  # under INDEX_DIRNAME (.venice) for the project tier
MEMORY_FILENAME = "memory.json"
TASKS_FILENAME = "tasks.json"

# Opt-in verbatim dump of each API response's `usage` block to stderr (#98). The
# ledger aggregates, and an aggregate cannot answer "was the field even there?"
# after the fact -- nothing persists the raw block anywhere (`venice code --json`
# and the session envelope both carry only the summed `CostLedger.to_dict()`), so
# diagnosing a cache regression meant re-running the incident live. Diagnostic
# only: stderr, so a piped stdout stays machine-readable.
ENV_USAGE_RAW = "VENICE_USAGE_RAW"

SFX_POLL_INTERVAL_SEC = 2.0
SFX_POLL_MAX_WAIT_SEC = 300

# `venice music` rides the same /audio/* queue as sfx and has always borrowed
# its cadence. Aliased rather than borrowed by name so music's call sites read
# honestly next to `defaults.music.poll_interval`, and so diverging the two is a
# one-line change here instead of a fan-out. (#57 Class C2)
MUSIC_POLL_INTERVAL_SEC = SFX_POLL_INTERVAL_SEC
MUSIC_POLL_MAX_WAIT_SEC = SFX_POLL_MAX_WAIT_SEC

# Video generation runs minutes, not seconds -- poll less often, wait longer.
VIDEO_POLL_INTERVAL_SEC = 5.0
VIDEO_POLL_MAX_WAIT_SEC = 900
