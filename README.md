# venice-cli

A Python CLI wrapping the [Venice.ai](https://venice.ai) API. The base is
stdlib-only; the optional `venice chat` and `venice embed` commands use the
official OpenAI SDK (Venice is OpenAI-compatible).

```sh
pip install venice-cli
```

> **Unofficial.** This is an independent, community-maintained client. It is not
> affiliated with, endorsed by, or supported by Venice.ai. "Venice" and
> "Venice.ai" belong to their respective owners. For official support, see
> [venice.ai](https://venice.ai).

Ships working `venice login`, `venice sfx` (sound-effect generation),
`venice music` (long-form ambience/music), `venice video` (text-to-video),
`venice tts` (text-to-speech), `venice image` (image generation),
`venice upscale` / `venice bg-remove` (image post-processing),
`venice master` (audio mastering), `venice contact-sheet` (montage grids of
generated images), `venice chat` (one-shot chat completions with Venice
extensions), `venice embed` (text embeddings), `venice balance` (budget
tracking), and `venice models` (catalog browser).

## Install

```sh
pip install venice-cli              # base: stdlib-only, no dependencies
pip install "venice-cli[openai]"    # + venice chat / venice embed
```

The distribution is named `venice-cli`, but the command is `venice` (and the
import package is `venice`). `pipx install "venice-cli[openai]"` works too, and
keeps the CLI out of your system site-packages.

### Dependencies

The base install pulls in **nothing** — every command is stdlib-only except
`venice chat` and `venice embed`, which use the official OpenAI SDK against
Venice's OpenAI-compatible API, and `venice mcp-serve`, which uses the MCP SDK.
Those SDKs are lazy-imported, so they ship as optional extras rather than hard
requirements: if you only generate images or audio, you don't pay for them.
Without the relevant extra, that command exits 2 with a hint; every other
command works normally.

Extras are per-feature and additive, so the pattern holds as the CLI grows:

| Install | Enables |
| --- | --- |
| `venice-cli` | everything except chat/embed and the MCP server |
| `venice-cli[openai]` | `venice chat`, `venice embed` |
| `venice-cli[mcp]` | `venice mcp-serve` (MCP stdio server; needs Python ≥ 3.10) |
| `venice-cli[all]` | every extra (`openai` + `mcp`) |

The `[mcp]` extra pulls in the [`mcp`](https://pypi.org/project/mcp/) SDK, which
requires Python ≥ 3.10. The base CLI still supports 3.9 — on 3.9 the extra
resolves to nothing and only `venice mcp-serve` is unavailable.

Some commands shell out to external binaries when present: `venice master` and
`venice contact-sheet` use `ffmpeg`/`ffprobe` (and ImageMagick's `montage` if
available); audio playback uses `mpg123`, `ffplay`, or `paplay`. These are
detected at runtime — nothing breaks if they're missing.

### From source (development)

Clone anywhere; no install is needed to run it:

```sh
git clone https://github.com/gobha-me/venice-cli.git
cd venice-cli
PYTHONPATH=src python3 -m venice --help
```

For an editable install: `pip install -e ".[openai]"`.

Alternatively `./install.sh` puts `venice` on your PATH without pip, by creating
two symlinks:
- `~/.local/bin/venice` -> `<repo>/bin/venice`
- `~/.local/lib/venice` -> `<repo>/src/venice`

and `~/.config/venice/` (mode 0700) for the credentials file. The installer
resolves the repo path itself, so the clone can live wherever you like.
`~/.local/bin` should be on your PATH.

> **Don't mix pip and `./install.sh`.** Both own `~/.local/bin/venice`. If pip
> got there first, `install.sh` refuses to clobber the real file and exits 1. If
> `install.sh` got there first, pip silently replaces the symlink — your repo
> edits stop taking effect with no error. Pick one; `pip uninstall venice-cli`
> or `./uninstall.sh` to back the other out.

## First-time setup

```sh
venice login
```

You'll be prompted (hidden input) for your API key from
<https://venice.ai/settings/api>. The key is stored at
`~/.config/venice/credentials` with mode 0600.

`$VENICE_API_KEY` in the environment overrides the file.

## Balance and budget tracking

Venice has up to four spendable buckets, drained in this order:

1. **DIEM allowance** — daily credit derived from staked DIEM tokens
   (1 DIEM staked = $1/day). Resets every 24h at `nextEpochBegins`.
   Per-epoch use-it-or-lose-it.
2. **Monthly credit** (BUNDLED_CREDITS) — bundle granted with paid
   subscriptions; drains before cash.
3. **VCU** — Venice Compute Units, per-tier inclusions.
4. **USD cash** — one-and-done prepaid USD balance.

1 unit of any bucket == $1 of purchasing power (per-model pricing in
`/models` lists the same number in both `usd` and `diem` fields).

**Inference-key visibility**: this CLI reads `USD` + `DIEM` and the
epoch reset time. The monthly-bundle and VCU balances live behind
admin-key endpoints (`/billing/balance`, 401 with inference keys), so
the CLI documents them but can't show their values. If you have
monthly credit, the actual debit lands there before USD cash, so the
"After charge" line on the USD cash side may be slightly pessimistic.

```sh
venice balance              # -> $32.70 USD   (combined visible total)
venice balance --verbose    # buckets, epoch reset, spend order
venice balance --json       # incl. total_usd_equiv, spend_order, notes
venice balance --min 5      # exit 1 if total < $5 (useful in scripts)
```

`venice sfx` and `venice tts` print the balance line inline next to the
cost quote, with DIEM listed first to mirror the drain order:

```
Balance:        $32.70 USD (6.56 DIEM allowance + 26.14 USD cash)
After charge:   $32.69 USD
```

Suppress with `--no-balance`. Hard-cap a single call with
`--max-spend USD` (refuses to queue / synthesize if the estimate
exceeds the cap).

## Sound effects

```sh
# Quote only -- no charge, no audio. Shows balance + estimated remaining.
venice sfx "thunderstorm rolling in" --duration 8 --dry-run

# Generate, confirm cost, save to ./venice-sfx-<id>.mp3.
venice sfx "soft chime" --duration 2

# Auto-confirm, custom output path, no playback, hard budget cap.
venice sfx "rain on tin roof" --duration 4 --yes --max-spend 0.05 \
    -o /tmp/rain.mp3 --no-play

# Background: prints queue_id to stdout, fetch later.
ID=$(venice sfx "ocean waves" --duration 10 --yes --background)
venice sfx-status "$ID" -o /tmp/ocean.mp3
```

### SFX Models

| slug | max duration |
|---|---|
| `elevenlabs-sound-effects-v2` (default) | 22 s |
| `mmaudio-v2-text-to-audio` | 30 s |

Durations longer than the model max are clamped (warning on stderr).

## Text-to-speech

```sh
# Positional text, auto-confirm, sub-cent cap.
venice tts "Hello from Venice." --yes --max-spend 0.01

# Read input from a file.
venice tts --from-file speech.txt --yes -o out.mp3

# Read input from stdin (pipe-friendly).
cat speech.txt | venice tts --stdin --yes -o out.mp3
echo "quick line" | venice tts --stdin --yes

# Specific voice and WAV output.
venice tts "Sky voice in wav." --voice af_sky --format wav --yes -o sky.wav

# Different model (e.g. ElevenLabs Turbo for higher quality).
venice tts "Demo line." --model tts-elevenlabs-turbo-v2-5 --voice <id> --yes

# Speed control (0.25-4.0).
venice tts "Fast talker." --speed 1.4 --yes

# Dry-run shows estimated cost + balance without spending.
venice tts "How much will this cost?" --dry-run
```

### TTS models and pricing (per 1M characters)

| slug | price | voices |
|---|---|---|
| `tts-kokoro` (default) | $3.50 | 54 |
| `tts-inworld-1-5-max` | $12.50 | 14 |
| `tts-xai-v1` | $18.75 | 5 |
| `tts-chatterbox-hd` | $50.00 | 9 |
| `tts-orpheus` | $62.50 | 8 |
| `tts-elevenlabs-turbo-v2-5` | $62.50 | 21 |
| `tts-qwen3-0-6b` | $87.50 | 9 |
| `tts-qwen3-1-7b` | $112.50 | 9 |
| `tts-minimax-speech-02-hd` | $125.00 | 15 |
| `tts-gemini-3-1-flash` | $187.50 | 30 |

To see the voice list for any TTS model:
```sh
venice models tts-kokoro | jq '.model_spec.voices'
```

If `--voice` is omitted Venice uses each model's built-in default.
Formats supported: `mp3` (default), `opus`, `aac`, `flac`, `wav`, `pcm`.

## Image generation

Sync `POST /image/generate`, same budget rails as `sfx`/`tts`. Default
output is **PNG** (lossless, good for card art and upscaling).

```sh
# Single image. Confirm cost, save to ./venice-image-<id>.png.
venice image "a fierce red dragon, trading-card art"

# Meaningful filename + auto-confirm + hard cap.
venice image "ancient stone golem" --name stone-golem --yes --max-spend 0.05

# Generate 4 variants to pick the best -> ...-1.png ... -4.png.
venice image "frost wyrm, splash art" --variants 4 --yes

# Sizing + tuning (pixel-based models take --width/--height, max 1280).
venice image "portrait card frame" --width 768 --height 1024 \
    --negative-prompt "text, watermark" --seed 42 --cfg-scale 7.5 --yes

# Omit the Venice watermark (best for finished card art).
venice image "frost wyrm, splash art" --hide-watermark --yes

# Don't blur flagged fantasy/battle art.
venice image "epic battle, dramatic" --no-safe-mode --yes

# Dry-run: estimate cost + balance, list planned files, spend nothing.
venice image "how much will this cost?" --dry-run

# Batch a whole card set from a file (one prompt per line;
# optional 'name<TAB>prompt'; blank lines and '#' comments skipped).
venice image --from-file cards.tsv --yes -o ./card-art/
venice image --from-file cards.tsv --variants 2 --dry-run

# Shared look across a whole set: a style prefix prepended to every prompt
# plus one negative prompt applied to the entire batch. Output filenames
# stay based on each card's own prompt, not the prefix.
venice image --from-file cards.tsv -o ./card-art/ --yes \
    --style-prefix "dark fantasy oil painting, dramatic cinematic lighting" \
    --negative-prompt "text, watermark, signature, blurry, lowres"
```

### Shared style templating

For a consistent set (e.g. a whole card deck), keep the long style + negative
strings in one place with a **preset** instead of retyping them:

```sh
venice image --from-file cards.tsv -o ./card-art/ --yes \
    --preset dark-fantasy --preset-file ./presets.json
```

`--preset-file` defaults to `~/.config/venice/image_presets.json`; point it at a
file in your project to version presets alongside the assets. Format:

```json
{
  "dark-fantasy": {
    "style_prefix": "dark fantasy oil painting, dramatic cinematic lighting",
    "negative_prompt": "text, watermark, signature, blurry, lowres"
  }
}
```

Precedence: an explicit `--style-prefix` / `--negative-prompt` on the command
line overrides the preset; the preset fills whatever you leave off. A single
`--negative-prompt` (or the preset's) applies to every image in a `--from-file`
batch.

`cards.tsv` example (tab between name and prompt):

```
fire-dragon	A fierce red dragon breathing flame, trading-card art
stone-golem	An ancient moss-covered stone golem, trading-card art
An unnamed prompt gets a slug from its first few words
```

Choose a model with `--model` (default `venice-sd35`); see
`venice models --type image --detail` for ids and per-image pricing.
Formats: `png` (default), `webp`, `jpeg`. Aspect-ratio/resolution-tier
models take `--aspect-ratio`/`--resolution` instead of `--width`/`--height`.
`--hide-watermark` drops the Venice watermark (Venice may keep it for some
content); `--no-safe-mode` stops adult-classified art from being blurred.

## Upscale images

`venice image` caps output at 1280px, so take art larger by upscaling it
(1-4x, default 2x) via `/image/upscale`:

```sh
# 2x upscale -> ./env-upscaled.png (960x540 -> 1920x1080).
venice upscale env.png --scale 2 --yes

# Enhance-only pass (scale 1 requires --enhance) with a style hint.
venice upscale portrait.png --scale 1 --enhance --enhance-prompt gold --yes

# Custom output, tune how much the enhancer may change the image.
venice upscale card.png --scale 4 --enhance --enhance-creativity 0.3 \
    -o card-4k.png --yes

# Dry-run: show the planned output + balance, spend nothing.
venice upscale env.png --dry-run
```

Input is a PNG/JPEG file under 25 MB. Pricing is **dynamic** (Venice bills
$0.001-$10.00 per call by input size and scale), so there's no reliable
pre-charge estimate; the balance is shown and you confirm (or `--yes`).

## Remove backgrounds

Venice's generate call ignores `background: transparent`, so make an asset
opaque then strip its background via `/image/background-remove` for a
transparent PNG (e.g. rank insignia, icons):

```sh
# Local file -> ./insignia-nobg.png (transparent).
venice bg-remove insignia.png --yes

# From a URL instead of a local file.
venice bg-remove --image-url https://example.com/logo.png -o logo-nobg.png --yes

# Dry-run: show the planned output + balance, spend nothing.
venice bg-remove insignia.png --dry-run
```

Provide exactly one source: a positional file (base64-encoded under 25 MB) or
`--image-url`. Pricing is dynamic like `upscale`; balance is shown and you
confirm before the charge.

## Master audio

Venice's audio queue returns a model-default container (sfx = mp3) and its
`/audio/speech` exposes no sample-rate/bit-depth control, so loudness
normalization, true-peak limiting, and seamless looping are done locally.
`venice master` shells out to **ffmpeg** (and **ffprobe** for `--loop`) — no
API call, no spend — to produce a WAV master (default 48kHz/24-bit) with
2-pass `loudnorm`:

```sh
# 48k/24-bit WAV master, LUFS -16 / true-peak -1 dBTP -> ./track.mastered.wav
venice master track.mp3

# Seamless-loop ambience (crossfade the tail into the head).
venice master ambience.mp3 --loop --loop-crossfade 3 -o ambience-loop.wav

# Tune targets / format.
venice master pad.wav --lufs -14 --true-peak -1.5 --bit-depth 16 --sample-rate 44100

# Show the ffmpeg commands without running them (works without ffmpeg installed).
venice master track.mp3 --dry-run
```

The same flags are available on `venice music` / `venice sfx` via `--master`,
which masters the generated file right after it's saved (writing a sibling
`*.mastered.wav`). Requires ffmpeg; if it's missing the command errors **before
spending** rather than after generating:

```sh
venice music "tense dungeon drone" --duration 60 --yes --master --loop
venice sfx "campfire crackle" --duration 8 --yes --master
```

Needs ffmpeg on PATH (`sudo apt install ffmpeg`). ffprobe (bundled with ffmpeg)
is required only for `--loop`.

## Video

Text-to-video on the same async queue as `sfx`/`music`
(`/video/quote` → `/video/queue` → `/video/retrieve` → `/video/complete`),
writing an mp4 to `venice-video-<id>.mp4`. Generation runs minutes, not
seconds, so it polls less often and waits longer by default (`--poll-interval`
5s, `--max-wait` 900s). `--model` defaults to the catalog's `default`-trait
video model; available durations, resolutions, and aspect ratios vary by model.

```sh
# Quote only -- no spend.
venice video "a koi pond at dawn, slow push-in" --dry-run

# Generate a 5s clip (default), confirm the spend, save to ./out.mp4.
venice video "a koi pond at dawn" --duration 5s --resolution 720p -o out.mp4 --yes

# Pick a model / aspect ratio; drop the generated audio track.
venice video "neon city flythrough" --model seedance-2-0-text-to-video \
  --aspect-ratio 16:9 --no-audio --yes

# Queue now, fetch later.
ID=$(venice video "storm clouds timelapse" --duration 10s --background)
venice video-status "$ID"
```

Some (VPS-backed) models return a presigned `download_url` at queue time and
stream nothing back from `/video/retrieve`; the CLI fetches the mp4 from that
URL transparently. When queued with `--background`, the URL is printed alongside
the queue id — pass it back via `venice video-status <id> --download-url <url>`.

## Chat

One-shot `POST /chat/completions` via the OpenAI SDK (see
[Optional dependency](#optional-dependency)). Streams the reply by default;
`--model` is validated against `/models?type=text` (a free GET) before the
paid call, and defaults to the catalog's `default`-trait text model.

```sh
# Simplest: message as an argument, streamed to stdout.
venice chat "Explain DIEM staking in one sentence."

# System prompt, explicit model, no streaming.
venice chat "Rewrite this as a haiku." --system "You are a poet." \
    --model llama-3.3-70b --no-stream

# Read the message from stdin (either form).
echo "Summarize this changelog." | venice chat -
git log --oneline -20 | venice chat - --system "Group these into release notes."

# Raw response object for scripting (forces --no-stream).
venice chat "ping" --json | jq '.choices[0].message.content'
```

### Venice extensions

Venice augments the OpenAI schema with a `venice_parameters` block; these flags
map onto it:

```sh
# Live web search + inline source citations (printed to stderr).
venice chat "What shipped in the latest Venice API update?" \
    --web-search on --web-citations

# Scrape URLs in the message via Firecrawl.
venice chat "Summarize https://venice.ai/blog" --web-scraping

# Talk to a public Venice character by its Public ID slug.
venice chat "Introduce yourself." --character venice

# Reasoning models: drop <think> blocks, or disable thinking entirely.
venice chat "Tricky logic puzzle..." --strip-thinking
venice chat "Just answer fast." --no-thinking

# Omit Venice's supplied system prompt (uncensored/raw behavior).
venice chat "..." --no-venice-system-prompt

# xAI native web+X search on grok models (extra ~$0.01/search).
venice chat "Latest posts about Venice?" --model grok-4-20 --x-search
```

| flag | effect |
|---|---|
| `--web-search {auto,on,off}` | Venice web search (default off) |
| `--web-citations` | cite web sources (with `--web-search`) |
| `--web-scraping` | Firecrawl-scrape URLs in the message |
| `--character SLUG` | use a public Venice character |
| `--no-venice-system-prompt` | omit Venice's supplied system prompt |
| `--strip-thinking` | strip `<think>` blocks (reasoning models) |
| `--no-thinking` | disable thinking (reasoning models) |
| `--x-search` | xAI web+X search (grok; extra ~$0.01/search) |

Chat pricing is dynamic (per token, model-dependent), so there's no pre-call
quote; pass `--json` or watch the `usage:` line on stderr to see token counts.

### Agent / tool calling

With `--tools` (alias `--agent`), `venice chat` becomes a **self-contained agent**:
the model can call venice's own endpoints as in-process function tools and the
completion runs in a loop (model → tool call → tool result → repeat) until it
produces a final answer. These are the same seven capabilities `venice mcp-serve`
exposes — but here they run **in-process on the `[openai]` extra alone** (no `mcp`
SDK, no subprocess):

`venice_image`, `venice_tts`, `venice_sfx`, `venice_music`, `venice_upscale`,
`venice_bg_remove`, and `venice_chat` (a sub-completion / subagent primitive).

```sh
# One command, multiple steps: the model generates an image, then critiques it.
venice chat --tools "Generate a fire-elemental trading card, then critique it."

# Text-only agentic reasoning via the venice_chat subagent tool (no paid media).
venice chat --tools "Use venice_chat to draft a haiku, then improve it."

# Restrict the toolset and cap the number of tool calls.
venice chat --tools --tool venice_image --max-tool-calls 3 "Draw three logo ideas."
```

Details and safety:

- **Capability guard.** Tools are offered only if the chosen model advertises
  `supportsFunctionCalling`; on a non-tool model the command prints a note and
  degrades to a plain one-shot chat. Without `--tools`, `venice chat` is unchanged.
- **Spend gating** (paid tools) reuses the MCP rails: each paid call auto-approves
  under a per-call cap (`--max-spend`, default `$0.10` / `$VENICE_MCP_MAX_SPEND`).
  An over-cap call prompts `[y/N]` on a TTY; non-interactively (or if you decline)
  the block is handed back to the model, which adapts. `--yes` auto-approves every
  paid call (this bypasses the per-call cap — `--max-tool-calls` still bounds the
  count). The model itself can never raise its spending authority.
- `--output DIR` sets where generated files are written (default: cwd).
- **Non-streamed in v1.** The tool path buffers each turn, so `--stream` is ignored
  when `--tools` is on; `--json` prints the final completion object.

| flag | effect |
|---|---|
| `--tools` / `--agent` | enable the in-process tool-calling loop |
| `--tool NAME` | restrict to this tool (repeatable; default: all seven) |
| `--max-tool-calls N` | cap tool invocations before forcing an answer (default 8) |
| `--max-spend USD` | per-call auto-approve cap for paid tools |
| `--yes` / `-y` | auto-approve every paid tool call |
| `--output DIR` / `-o` | directory for generated files |

## Embeddings

Turn text into embedding vectors with a Venice embedding model (`/embeddings`,
via the OpenAI SDK). The model defaults to the catalog's `default`-trait
embedding model; pass `--model` to pick another (see
`venice models --type embedding`).

```sh
# Single input -> one JSON array on stdout.
venice embed "the quick brown fox"

# Read the input from stdin.
echo "summarize me" | venice embed -

# Batch: one input per non-empty line -> one vector per line (index order).
venice embed --from-file corpus.txt

# Pipe vectors to jq (newline-delimited JSON, one array per line).
venice embed "hello" | jq 'length'

# Truncate dimensions (if the model supports it) and pick a model.
venice embed "hello" --model text-embedding-qwen3-8b --dimensions 256

# Full raw response object (model, data, usage) instead of bare vectors.
venice embed "hello" --json | jq '.usage'
```

By default each embedding prints as a JSON array, one per line;
`--encoding-format base64` requests base64-packed vectors instead of floats.

## MCP server

`venice mcp-serve` runs an [MCP](https://modelcontextprotocol.io) server over
stdio, exposing venice's generators as tools that an MCP host (Claude Code, or
any MCP client) can call directly instead of shelling out to the CLI. It needs
the `[mcp]` extra (Python ≥ 3.10):

```sh
pip install "venice-cli[mcp]"

# Register it with Claude Code:
claude mcp add venice -- venice mcp-serve
```

The server exposes seven tools:

| Tool | Does | Paid? |
| --- | --- | --- |
| `venice_image` | generate image(s) → file path(s) | yes (estimated) |
| `venice_tts` | synthesize speech → audio file | yes (estimated) |
| `venice_sfx` | sound effect (async queue) → audio file | yes (quoted) |
| `venice_music` | long-form music/ambience (async queue) → audio file | yes (quoted) |
| `venice_upscale` | upscale/enhance a local image → image file | yes (dynamic) |
| `venice_bg_remove` | remove a background → transparent PNG | yes (dynamic) |
| `venice_chat` | one-shot chat completion → reply text | no |

**Spend gating.** MCP is non-interactive, so instead of a `[y/N]` prompt the
paid tools gate on cost. A tool call whose estimated cost is at or under the
auto-approve cap (`VENICE_MCP_MAX_SPEND`, default **$0.10**) runs immediately.
If the estimate is over the cap — or can't be known up front, as with the
dynamically-priced `venice_upscale` / `venice_bg_remove` — the tool returns
`{"status": "confirmation_required", ...}` with the estimate and cap, and the
host must re-call with `confirm=true` (or a higher `max_spend`). Nothing is
spent and no file is written on a gated call. `venice_chat` is cheap and not
gated.

**Output.** Tools write their result to a file and return its **path** (never
inline base64). Files land in `VENICE_MCP_OUTPUT_DIR` (default: the current
working directory), or a per-call `output_dir`. The API key is read the usual
way (`$VENICE_API_KEY` or the credentials file) and is never echoed.

Only stdout carries the JSON-RPC protocol; the server's own diagnostics go to
stderr. Video and image-edit tools are not exposed yet (tracked separately).

## Config

`venice config` manages a persistent, non-secret config file at
`~/.config/venice/config.json` (created mode 0600). It holds two things: an
**MCP server registry** (for a future `venice chat --mcp` host) and **default
flag values** so you stop repeating `--model` / `-o` / `--yes` / `--max-spend`.

```sh
# MCP server registry (like `claude mcp add`)
venice config add venice --command venice --arg mcp-serve      # stdio server
venice config add remote --url https://host/mcp --header 'Authorization: Bearer T'
venice config list
venice config show [NAME]
venice config remove venice

# Default flag values (dotted keys)
venice config set defaults.chat.model llama-3.3-70b
venice config set defaults.max_spend 0.50
venice config get defaults.chat.model
venice config unset defaults.chat.model
```

The file looks like:

```json
{
  "version": 1,
  "mcpServers": {
    "venice": { "command": "venice", "args": ["mcp-serve"] }
  },
  "defaults": {
    "output_dir": "~/venice-out",
    "max_spend": 0.50,
    "chat": { "model": "llama-3.3-70b", "web_search": "auto" }
  }
}
```

Global keys under `defaults` (`output_dir`, `max_spend`, `yes`) apply to any
command that has the flag; a per-command section (e.g. `defaults.chat`)
overrides them. **Precedence for any flag is: explicit CLI flag > environment
variable > config file > built-in default** — so a config default never shadows
something you pass on the command line or set in the environment.

The **API key is never stored here** — it stays in
`~/.config/venice/credentials`. Unknown keys are preserved on write, so the
schema is forward-compatible.

## Browse the model catalog

```sh
venice models                          # count by type
venice models --type music             # list ids, one per line
venice models --type music --detail    # ids + name + pricing + capabilities
venice models elevenlabs-sound-effects-v2   # full JSON for one model
venice models --type all --json        # everything, raw
```

At time of writing the catalog spans ~258 models across text (80),
code (30), image (26), **video (92)**, music+sfx (10), tts (10),
embedding (9), and upscale (1). The video models include Sora 2,
Veo 3.1, Kling, Runway Gen4, LTX-2, Wan 2.7, Seedance 2.0, and more.

### Exit codes

| exit | meaning |
|---|---|
| 0 | success |
| 1 | user declined / aborted / insufficient balance (402) |
| 2 | bad input, no API key, missing prompt, stub command |
| 3 | content policy block (422) |
| 4 | rate limit (429) |
| 5 | Venice 5xx |
| 6 | job not found / expired (404) |
| 7 | poll timeout |
| 8 | network / connection error |
| 9 | disk write error |
| 130 | Ctrl-C |

## Audio playback caveat

Auto-play depends on whatever player your system has. If all you have
is `paplay`, note that it plays WAV natively but handles MP3 (Venice's
default output) via PulseAudio's GStreamer plumbing, which can fail
silently. If playback fails the file is still saved; the CLI just
won't auto-play it. For reliable MP3 playback in-CLI, install one of:

```sh
sudo apt install mpg123    # or: ffmpeg
```

The player list (`paplay` -> `aplay` -> `ffplay` -> `mpg123` -> `play`
-> `afplay`) auto-picks the first available.

## Environment overrides

| var | meaning |
|---|---|
| `VENICE_API_KEY` | overrides the file-based key (no disk read) |
| `VENICE_BASE_URL` | override the API base URL (testing, proxy) |
| `VENICE_MCP_MAX_SPEND` | `mcp-serve` auto-approve cap in USD (default `0.10`) |
| `VENICE_MCP_OUTPUT_DIR` | where `mcp-serve` tools write files (default: cwd) |

## Commands at a glance

| command | does what |
|---|---|
| `venice login` | store API key (interactive, hidden input, mode 0600) |
| `venice balance [--verbose\|--json\|--min N]` | current USD + DIEM balance |
| `venice models [--type T] [--detail] [SLUG]` | browse the catalog |
| `venice sfx PROMPT [--duration N] [--max-spend USD] [...]` | generate a sound effect |
| `venice sfx-status QUEUE_ID` | fetch a backgrounded SFX job |
| `venice tts TEXT [--voice V] [--format F] [--speed N] [...]` | synthesize speech (sync) |
| `venice image PROMPT [--variants N] [--name NAME] [--max-spend USD] [...]` | generate image(s) (sync) |
| `venice image --from-file PATH [...]` | batch-generate a card set |
| `venice music PROMPT [--duration N] [--master] [--loop] [...]` | generate long-form ambience/music |
| `venice video PROMPT [--duration 5s] [--resolution R] [--aspect-ratio A] [...]` | generate a video (async queue, mp4) |
| `venice video-status QUEUE_ID [--download-url URL]` | fetch a backgrounded video job |
| `venice master INPUT [--loop] [--lufs N] [--bit-depth N] [...]` | master audio to WAV (48k/24-bit, LUFS/true-peak) |
| `venice contact-sheet DIR_OR_GLOB [--cols N] [--cell WxH] [--label] [...]` | tile images into one contact sheet (no API call) |
| `venice chat MESSAGE [--system S] [--model M] [--web-search on] [...]` | one-shot chat completion (OpenAI SDK) |
| `venice embed [TEXT] [--from-file PATH] [--model M] [--dimensions N] [--json]` | text embeddings (OpenAI SDK) |
| `venice mcp-serve` | run an MCP server (stdio) exposing venice tools (needs `[mcp]`) |
| `venice config add\|list\|remove\|show` | manage the MCP server registry |
| `venice config get\|set\|unset KEY [VALUE]` | manage default flag values |

## Tests

```sh
make test
```

Stdlib `unittest` only. Tests mock `urlopen` (and, for `chat`, the OpenAI
client) and patch `HOME` to a tmpdir -- no live API calls, no real disk writes
outside the tmpdir. The `chat` and `embed` tests need the OpenAI SDK importable
(`pip install -e ".[openai]"`).

## Uninstall

If you installed with pip:

```sh
pip uninstall venice-cli
```

If you installed from source with `./install.sh`:

```sh
./uninstall.sh
```

Either way the credentials file at `~/.config/venice/credentials` is left
alone -- delete it manually if you want. `uninstall.sh` removes only the two
symlinks, and only if they point at that repo.

## Security note

The API key is stored **plaintext on disk** at
`~/.config/venice/credentials` (mode 0600, inside a 0700 directory).
There is no OS keychain integration -- file permissions are the only
protection, so anything that can read your home directory can read the
key. The `venice config` file (`~/.config/venice/config.json`) is written
mode 0600 for the same reason -- an MCP `env`/`headers` entry can carry a
bearer token -- but the API key itself is never written there.

- In CI or any shared environment, prefer `$VENICE_API_KEY` (it
  overrides the file) sourced from that system's secret store, and
  don't run `venice login` there.
- The key is never logged or printed by this tool, but it is visible to
  anything reading your process environment or scrollback. Be aware of
  what's on screen when sharing a terminal.
- If a key is exposed, revoke and rotate it at
  <https://venice.ai/settings/api>.
