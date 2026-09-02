# venice-cli

A Python CLI wrapping the [Venice.ai](https://venice.ai) API. The base package
has no required third-party dependencies; model-backed commands use the
optional OpenAI SDK (Venice is OpenAI-compatible), and MCP transport uses the
optional MCP SDK.

```sh
pip install venice-cli
```

> **Unofficial.** This is an independent, community-maintained client. It is not
> affiliated with, endorsed by, or supported by Venice.ai. "Venice" and
> "Venice.ai" belong to their respective owners. For official support, see
> [venice.ai](https://venice.ai).

Ships working `venice login`, `venice sfx` (sound-effect generation),
`venice music` (long-form ambience/music), `venice video` (text/image-to-video),
`venice tts` (text-to-speech), `venice image` (image generation),
`venice upscale` / `venice bg-remove` (image post-processing),
`venice master` (audio mastering), `venice contact-sheet` (montage grids of
generated images), `venice chat` (one-shot or interactive chat completions with
Venice extensions), `venice embed` (text embeddings), `venice index` /
`venice search` (project semantic search), `venice review` (cold-context code
review of a diff), `venice balance` (budget tracking), and `venice models`
(catalog browser).

## Install

```sh
pip install venice-cli              # base: stdlib-only, no dependencies
pip install "venice-cli[openai]"    # + model-backed chat/code/review/search features
```

The distribution is named `venice-cli`, but the command is `venice` (and the
import package is `venice`). `pipx install "venice-cli[openai]"` works too, and
keeps the CLI out of your system site-packages.

### Dependencies

The base install pulls in **nothing**: its package metadata has no required
third-party dependencies, and the complete command graph remains importable.
Some commands and features need an optional SDK when invoked. Those SDKs are
lazy-imported, so image/audio/video generation and the other stdlib-backed
features remain usable without them. A missing SDK fails at the feature boundary
with an installation hint rather than breaking the base CLI.

#### OpenAI feature inventory

This is the authoritative inventory of features that use the `[openai]` extra.
The name at the left is the corresponding lazy-import label in the source.

<!-- openai-extra-inventory:start -->
- `chat`: `venice chat`, including its agent loop and the MCP server's
  `venice_chat` tool. `venice chat --mcp` needs both `[openai]` for chat and
  `[mcp]` to attach external servers.
- `code`: `venice code` and its model-backed agent loop.
- `embed`: `venice embed`.
- `index`: `venice index` and the agent's `reindex` tool.
- `search`: `venice search` and the agent's `project_search` tool.
- `review`: model-backed `venice review` runs. A diff skipped by `auto` triage,
  an empty diff, or `--effort never` returns before importing the SDK.
- `vision`: the agent's model-backed `vision` tool.
- `mcp-serve --http`: the authenticated remote MCP server's `venice_chat` and
  delegated `venice_vision` tools.
<!-- openai-extra-inventory:end -->

The `[mcp]` extra is independent: it provides the transport used to start
`venice mcp-serve` or attach servers with `venice chat --mcp`. Starting the
Venice MCP server needs only `[mcp]`; invoking its model-delegating
`venice_chat` tool additionally needs `[openai]`.

Extras are per-feature and additive, so the pattern holds as the CLI grows:

| Install | Enables |
| --- | --- |
| `venice-cli` | dependency-free command graph and stdlib-backed features; SDK-backed features above remain unavailable |
| `venice-cli[openai]` | every model-backed feature in the authoritative inventory above |
| `venice-cli[mcp]` | MCP server/client transport; needs Python ≥ 3.10, and does not itself provide `[openai]` |
| `venice-cli[all]` | every extra (`openai` + `mcp`) |

The `[mcp]` extra pulls in version 2.x of the
[`mcp`](https://pypi.org/project/mcp/) SDK, which requires Python ≥ 3.10. The
base CLI still supports 3.9 — on 3.9 the extra
resolves to nothing and only `venice mcp-serve` and `venice chat --mcp` are
unavailable.

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
one symlink:
- `~/.local/bin/venice` -> `<repo>/bin/venice`

and `~/.config/venice/` (mode 0700) for the credentials file. The installer
resolves the repo path itself, so the clone can live wherever you like.
`~/.local/bin` should be on your PATH.

> **Don't mix pip and `./install.sh`.** Both own `~/.local/bin/venice`. If pip
> got there first, `install.sh` identifies the pip wrapper, refuses to clobber
> it, and prints the uninstall/editable-install choices. If `install.sh` got
> there first, pip silently replaces the symlink — your repo edits stop taking
> effect with no error. Pick one; `pip uninstall venice-cli` or `./uninstall.sh`
> to back the other out.

### Shell completion

`venice completion [bash|zsh]` prints a tab-completion script to stdout, generated
by introspecting the CLI so it never drifts from the real commands and flags:

```sh
# bash -- current shell:
source <(venice completion bash)
# bash -- persistent (per-user):
venice completion bash > ~/.local/share/bash-completion/completions/venice

# zsh -- drop it somewhere on your $fpath, named `_venice`:
venice completion zsh > "${fpath[1]}/_venice"   # then restart zsh
```

`./install.sh` writes the bash script for you (source installs only); a
`pip install` uses the `source <(...)` line above. Completion covers subcommands,
the `config`/`secret` nested actions (with their aliases), and each command's
flags.

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

Suppress with `--no-balance`, or set `defaults.no_balance` once to suppress it
everywhere (`--show-balance` re-enables it for a single run). Hard-cap a single call
with `--max-spend USD` (refuses to queue / synthesize if the estimate exceeds
the cap). Caps and polling durations must be finite numbers; `NaN` and infinity
are rejected before any paid request. When an endpoint has no reliable upfront
estimate, supplying `--max-spend` fails closed instead of treating the cap as
unmetered; use the ordinary confirmation prompt or `--yes` without a cap if you
accept that endpoint's dynamic price.

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

### TTS models, formats, and pricing

Models, voices, formats, defaults, and prices are resolved from the live TTS
catalog so newly added models do not require a CLI release. To inspect one model:

```sh
venice models tts-kokoro | jq '.model_spec | {voices, supported_formats, default_format, pricing}'
```

If `--voice` is omitted Venice uses each model's built-in default.
If `--format` is omitted Venice uses that model's advertised `default_format`;
an explicit format is validated against its `supported_formats` before confirmation.

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

# Model-native controls. Style references accept local files, HTTP(S) URLs,
# data URLs, or raw base64; repeat the JSON flag for each reference.
venice image "portrait in the same painted style" --resolution 1K --quality high \
    --style-reference '{"image":"style.png","strength":0.7}' \
    --embed-exif-metadata --enhance-prompt --yes

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
Native controls include repeatable `--style-reference JSON`, `--quality`,
`--lora-strength`, `--embed-exif-metadata`, `--web-search`,
`--disable-prompt-optimization-thinking`, and `--enhance-prompt` (with explicit
inverse forms for boolean options). Style-reference image data is bounded below
8 MiB; local files are checked as recognized images and encoded as data URLs.
Raw base64 is accepted and preserved. Reference count,
reference-strength support, and quality are validated against the selected
model's live catalog entry. Quality/resolution pricing is selected from the
matching catalog matrix. Because web search and prompt enhancement can add an
unlisted charge, their total is treated as unknown: `--max-spend` fails closed
and MCP/agent callers must explicitly confirm.
`--hide-watermark` drops the Venice watermark (Venice may keep it for some
content); `--no-safe-mode` stops adult-classified art from being blurred. To
drop the watermark **by default**, set `defaults.image.hide_watermark` in config
(`venice config set defaults.image.hide_watermark true`); `--no-hide-watermark`
forces it back on for a single call.

## Upscale images

`venice image` caps output at 1280px, so take art larger by upscaling it
(2x or 4x, default 2x) via `/image/upscale`:

```sh
# 2x upscale -> ./env-upscaled.png (960x540 -> 1920x1080).
venice upscale env.png --scale 2 --yes

# Custom output; creativity controls added detail/texture (0-0.02).
venice upscale card.png --scale 4 --creativity 0.01 \
    -o card-4k.png --yes

# Dry-run: show the planned output + balance, spend nothing.
venice upscale env.png --dry-run
```

Input is a PNG/JPEG file under 25 MB. Venice's retired enhancer controls are no
longer accepted; the current request contains only the image, scale, and optional
creativity. The upstream contract changed after the original #1 workflow. Older
configs containing `defaults.upscale.enhance`, `enhance_creativity`,
`enhance_prompt`, or `replication` fail closed with `venice config unset` cleanup
guidance instead of silently changing a paid request. Pricing is **dynamic** (Venice bills
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

## Edit images

Iterate on already-generated art without regenerating it — recolor, restyle, or
inpaint a card — via `/image/edit`. Add repeatable `--layer` images (masks or
overlays) to composite instead, which routes to `/image/multi-edit` (base
first). The selected model's live `maxInputImages` constraint supplies the
limit:

```sh
# Prompt-only edit -> ./card-edit.png.
venice image-edit card.png -p "change the sky to a sunrise" --yes

# From a URL, request a 16:9 JPEG at 2K.
venice image-edit --image-url https://example.com/card.png \
    -p "make it snow" --aspect-ratio 16:9 --output-format jpeg \
    --resolution 2K -o card-winter.jpg --yes

# Mask/overlay composite -> /image/multi-edit (base first, then layers).
venice image-edit base.png -p "apply this mask" --layer mask.png --yes

# Multi-edit-only quality plus prompt controls shared by both edit routes.
venice image-edit base.png -p "combine these references" --layer style.png \
    --quality high --enhance-prompt \
    --disable-prompt-optimization-thinking --yes

# Dry-run: show the planned output + balance, spend nothing.
venice image-edit card.png -p "brighter" --dry-run
```

Provide exactly one base source: a positional file (base64-encoded under 25 MB)
or `--image-url`. `--prompt/-p` is required. Optional `--model`, `--aspect-ratio`,
`--resolution`, `--output-format`, prompt optimization controls, and
`--safe-mode`/`--no-safe-mode` map straight to the API. `--quality` applies only
to multi-edit and is checked against that model's advertised qualities. These
preferences are config-backable via `defaults.image_edit.*`;
omit them to take the model defaults (`firered-image-edit`, PNG for 1K). Pricing
is dynamic like `upscale`; balance is shown and you confirm before the charge.

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
`*.mastered.wav`). Because one chain is shared by all three commands, the five
target knobs are **global** config keys rather than per-command ones —
`venice config set defaults.lufs -14` retargets `master`, `sfx --master` and
`music --master` together, while `defaults.sfx.lufs` still overrides it for
`sfx` alone. Requires ffmpeg; if it's missing the command errors **before
spending** rather than after generating:

```sh
venice music "tense dungeon drone" --duration 60 --yes --master --loop
venice sfx "campfire crackle" --duration 8 --yes --master
```

Needs ffmpeg on PATH (`sudo apt install ffmpeg`). ffprobe (bundled with ffmpeg)
is required only for `--loop`. Both toggles are config-backable
(`defaults.sfx.master`, `defaults.music.master`, `defaults.{master,sfx,music}.loop`)
and both have `--no-master`/`--no-loop` to override a config default for one run.

## Video

Text-to-video (and image-to-video, see below) on the same async queue as
`sfx`/`music` (`/video/quote` → `/video/queue` → `/video/retrieve` →
`/video/complete`), writing an mp4 to `venice-video-<id>.mp4`. Generation runs
minutes, not seconds, so it polls less often and waits longer by default
(`--poll-interval` 5s, `--max-wait` 900s -- both config-backable via
`defaults.video.*`). `--model` defaults to the catalog's
`default`-trait video model. Duration, resolution, and aspect ratio are checked
against that selected model's live catalog constraints before the quote;
media-input support also varies by model.

```sh
# Quote only -- no spend.
venice video "a koi pond at dawn, slow push-in" --dry-run

# Generate a 5s clip (default), confirm the spend, save to ./out.mp4.
venice video "a koi pond at dawn" --duration 5s --resolution 720p -o out.mp4 --yes

# Pick a model / aspect ratio; drop the generated audio track.
# (`defaults.video.no_audio` makes that the default; `--with-audio` re-enables
# it for one run. Note `--audio` is different -- it supplies an INPUT audio file.)
venice video "neon city flythrough" --model seedance-2-0-text-to-video \
  --aspect-ratio 16:9 --no-audio --yes

# Queue now, fetch later.
ID=$(venice video "storm clouds timelapse" --duration 10s --background)
venice video-status "$ID"
```

### Media inputs (image-to-video & references)

For models that support them, the generation can be conditioned on media.
Every media flag accepts **a local file path or an `http(s)`/`data:` URL** —
local files are read, size-checked, and encoded to a `data:` URL for you.

```sh
# Image-to-video: animate a still (optionally with an end frame).
venice video "slow zoom out from the figure" --image hero.png --yes
venice video "morph A into B" --image a.png --end-image b.png --yes

# Reference images for character/style consistency (repeatable, up to 30).
venice video "the same knight, new scene" \
  --reference-image knight1.png --reference-image knight2.png --yes

# Video-to-video / upscale, and reference videos (repeatable, up to 10). The
# aggregate reference-video duration feeds the *quote* so R2V pricing is right.
venice video "restyle this clip" --video source.mp4 \
  --reference-video ref.mp4 --reference-video-duration 5 --yes

# Advanced @Element composition (Kling O3): pass each element as a JSON object.
# Local paths inside the JSON are encoded just like the flags above.
venice video "@Element1 greets @Element2 at @Image1" \
  --element '{"frontal_image_url":"alice.png"}' \
  --element '{"frontal_image_url":"bob.png"}' \
  --scene-image plaza.png --yes
```

Full media flags: `--image`, `--end-image`, `--video`, `--audio` (background
music, distinct from `--no-audio`), `--reference-image` (≤30),
`--reference-video` (≤10), `--reference-audio` (≤10), `--scene-image` (≤4),
`--reference-video-duration`, and `--element` (JSON, ≤4). Image/reference
inputs condition generation and are sent only on `/video/queue`; `--video` and
`--reference-video-duration` also reach `/video/quote` because they change the
price. Per-model support varies — the API rejects an unsupported combination.

Some (VPS-backed) models return a presigned `download_url` at queue time and
stream nothing back from `/video/retrieve`; the CLI fetches the mp4 from that
URL transparently. Background jobs bind that URL to the queue id in a private
mode-0600 local registry at `~/.config/venice/video_jobs.json`, so the signed URL
is not printed or passed through an agent/MCP transcript. Entries are capped at
100 and expire after seven days. `video-status --download-url` remains only as a
deprecated operator fallback for jobs queued before v0.83.10.

Presigned downloads use a fail-closed HTTPS boundary: every redirect is resolved
and connected through a validated, pinned public address while retaining normal
TLS hostname verification. Local, private, link-local, multicast, reserved,
unspecified, and metadata destinations are rejected for IPv4 and IPv6; ambient
proxies are ignored. Downloads accept video media types only, allow at most five
redirects, stream to a private temporary file, and stop at 512 MiB or 15 minutes.

## Chat

`POST /chat/completions` via the OpenAI SDK (see
[Optional dependency](#optional-dependency)) — one-shot, or an interactive
multi-turn REPL (see [Interactive mode](#interactive-mode)). Streams the reply by
default; `--model` is validated against `/models?type=text` (a free GET) before
the paid call, and defaults to the catalog's `default`-trait text model.

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

Plain human-readable chat reports each response's token counts on stderr as
`usage: prompt=N completion=N total=N`, for both streamed and `--no-stream`
calls. `--json` keeps stderr quiet because the raw response on stdout already
contains the same `usage` object. Tool-calling chat instead reports its distinct
whole-run time and cost footer, including under `--json`.

Ctrl+C during a one-shot chat prints `chat: aborted` and exits 130. Because the
default streamed form may already have written part of the reply to stdout, its
notice is `chat: aborted (partial output may appear above)`.

### Interactive mode

With `-i`/`--interactive` — or simply no message on a terminal — `venice chat`
drops into a REPL that holds the conversation in memory across turns. All the
Venice extensions and `--tools` (each turn becomes an
[agent](#agent--tool-calling) turn) carry over. Each REPL is a **session** that
auto-saves after every turn (see [Sessions](#sessions), below), so you can pick
it back up later — settings and all.

```sh
# Start a conversation (or just run `venice chat` on a TTY).
venice chat -i --system "You are a terse assistant."

# Resume the most recent chat session, or a specific one by id.
venice chat --continue
venice chat --resume 20260722T220353-9ab8e7

# Resume an old hand-saved transcript file (still works).
venice chat --resume session.json

# Don't persist this one.
venice chat -i --ephemeral
```

In-REPL slash-commands: `/system [text]` (show/set the system prompt),
`/persona [name]` (load a saved system prompt from
`~/.config/venice/personas/`; with no name, list the available ones),
`/model [name]` (switch model; with no name, show the current one and list the
catalog), `/models` (list the available models, marking the current and the
default), `/auto` and `/manual` (toggle auto-accepting paid/side-effecting tool
calls for following turns), `/compact [N]` (summarize older history into one
message, keeping the last `N` turns verbatim),
`/context list [CURSOR]` (list archived evidence metadata) and
`/context read ID [OFFSET]` (read up to 32 KiB of an exact archived message),
`/cost` (this session's estimated spend so far, with the active model's
current-run cache hit rate on the same line; `--session-max-spend` adds a
cap), `/usage` (a token + cost breakdown for the session, keeping the
cache-read/cache-write/uncached input split distinct so cache-heavy sessions
cost out correctly — the split and the hit rate report `n/a` when the model's
usage block carried no cache fields at all, so a printed `0.0%` always means the
provider reported a real zero rather than that nobody looked; that same rule
governs `/cost` and both run footers — plus a
**wall-clock** row — total, turn count and average
for the time the CLI kept you waiting, measured from submitting a turn to
getting the prompt back, so thinking time at the prompt is never counted; it
accumulates across `--resume` — plus a **per-API-call trace** with a row per model
call and a marker wherever compaction or a resume reseed moved the prefix, plus
cumulative and current-run rows per model — which keeps a model switch or resume
from averaging two unrelated cache histories together),
`/reset` (clear history, keep the system prompt),
`/save [file]` (write the transcript JSON; defaults to the `--resume` file),
`/paste` (compose a multi-line message a line at a time, ending with `/end` —
`/cancel` aborts), `/edit [text]` (compose your next message in `$EDITOR`, like
`git commit`, optionally pre-seeded with `text`),
`/help`, and `/exit` (or `/quit`, or Ctrl-D). Ctrl-C aborts the current turn
without ending the session (while a tool-loop turn is running it first pauses to
steer — see [Mid-run steering](#mid-run-steering-venice-sessions-send)); at the
prompt it just discards the half-typed line and re-prompts. However you leave —
`/exit`, Ctrl-D, or a Ctrl-C that lands somewhere unexpected — the session is
flushed to disk first, so slash-command edits like `/model` and `/system` are
never lost to the way you quit. Tab completes slash-commands (and model ids after
`/model `, persona names after `/persona `). At a per-tool confirmation prompt, `a` accepts that call **and**
auto-accepts the rest of the run. `--max-tool-calls 0` runs until the model
stops on its own (instead of capping at the default and asking to continue).

Long sessions can also compact automatically. `--auto-compact` (or
`defaults.chat.auto_compact` / `defaults.code.auto_compact`) summarizes the
older prefix into one synthetic message once the prompt crosses
`--compact-threshold` tokens (default 100 000), keeping the system prompt and
the last `--compact-keep-turns` turns (default 10) verbatim. The trigger uses
the server-reported `usage.prompt_tokens` when a turn provides it (falling
back to a chars-per-token estimate otherwise), so it fires on the real prompt
size rather than a guess. `--compact-loss-policy aggressive|evidence` controls
what happens to the summarized prefix. Chat defaults to `aggressive`, preserving
the original behavior. Code defaults to `evidence`: every removed user message,
assistant tool call (including arguments), and tool result is copied exactly into
the private session envelope before the live history is replaced. The model gets
a bounded index plus the read-only `venice_context_archive` tool; operators can use
the `/context` commands above even in a plain REPL. The archive is capped at 512
entries and 8 MiB. If the next compaction would cross either cap, it is refused
before the summary API call and both history and archive remain unchanged. Reads
are paged at 32 KiB and list pages at 50 entries. Compaction is otherwise
best-effort — a failed or empty summarization leaves both stores untouched — and
never orphans a tool result from its assistant turn. It's off by default because
it costs a summarization call.

#### Sessions

Every `venice chat` / `venice code` REPL is a **session** that auto-saves after
each turn to `~/.config/venice/sessions/<id>.json` (mode 0600;
`$VENICE_SESSIONS_DIR` overrides the location). Unlike a bare `/save` transcript,
a session travels with its **settings** — model, system prompt, generation
parameters, `max-tool-calls`, the `venice code` sandbox root, and the running
token/cost usage — plus any bounded exact context archive — so resuming restores
the whole context, not just the messages. Session envelope v2 adds
`context_archive` plus the structural count of authoritative leading system
messages; v1 envelopes remain readable, conservatively preserve their complete
leading system prefix, and resume with an empty archive.
The API key is never written to a session.

`venice code` also assigns each session an opaque `prompt_cache_key`, sent as an
OpenAI-compatible routing hint so successive plan, tool-loop, and verification calls
stay on cache-affine backend infrastructure when the provider supports it. The key is
not a credential. It survives `--resume` and `/reset`; disposable subagents receive
their own keys, while compaction's deliberately fresh summary request receives none.

One-shot `venice code "task"` runs are sessions too (they persist unless
`--ephemeral`), so an unattended `--auto` run is resumable, inspectable, and —
new in this release — **steerable while it runs** (see below).

```sh
venice sessions ls              # list saved sessions (newest first; flags pending steers)
venice sessions ls --json       # same, as JSON (adds each session's pending-steer count)
venice sessions show <id>       # settings + message summary for one session
venice sessions send <id> "…"   # queue a mid-run steering message (see below)
venice sessions rm <id>         # delete a session (and its steering mailbox)

venice chat --continue          # resume the most recent chat session
venice chat --resume <id>       # resume a specific session by id (restores settings)
venice code --continue          # (code's most recent session re-sandboxes to its root)
```

Resume precedence is **explicit flag > saved session > config default**: passing
e.g. `--model` on resume overrides the session's saved model, but omitting it
keeps what the session used. `--resume` still accepts a plain transcript **file**
for back-compat (it's imported into a fresh session, leaving the file untouched).
Pass `--ephemeral` (alias `--no-save`) to run without persisting a session (which
also makes the run unsteerable). `/save [file]` remains an explicit, separate
transcript export.

`venice code` rebuilds its root/tool-aware system prompt on resume. If those live
instructions differ from the stored prompt, it keeps the live sandbox accurate,
prints `code: resumed with a different system prompt; prompt cache will be cold`,
and records a `resume_reseed` marker in `usage.context_events`. Prompt text and
hashes are not copied into the usage record.

#### Mid-run steering (`venice sessions send`)

A running agent — especially `venice code --auto` — used to have only two
controls: let it finish, or kill it (losing uncommitted work and metered spend).
`venice sessions send` adds a third: **steer it without stopping it.**

```sh
venice sessions send latest "actually, prioritize the #3 regression first"
venice sessions send 20260724T101530-ab12cd "skip the CSS, focus on the API"
echo "long note…" | venice sessions send latest -     # read the message from stdin
```

The message is dropped into the session's file **mailbox**
(`~/.config/venice/sessions/<id>/mailbox/`, one atomic 0600 file per message).
At its next checkpoint — the boundary between tool calls, before the next model
turn — the agent drains the mailbox and consumes each message as a tagged user
turn, exactly as if you had typed it interactively. It's additive input, not a
reset: your `--max-tool-calls` / `--session-max-spend` budgets are unchanged.

`latest` targets the most recent `code` session (`sessions ls --json` shows ids +
pending counts for scripting). Targeting is **by recency, not liveness** — there's
no process tracking — so a message sent to a session that has already finished
simply waits in its mailbox and is drained the next time you `--resume` it. The
mailbox is a local, owner-only directory (0700), not a network channel: a steer
carries the same trust as the original task. This is stdlib-only — no daemon, no
sockets, no extra dependency.

When you're **watching** an attached run in a terminal (`venice code --interactive`,
or a foreground `--auto`/`--manual` run), you don't need a second shell — just press
**Ctrl+C**:

- **First Ctrl+C** lets the current step finish, then pauses at the next checkpoint and
  asks: `[paused] message to the agent (empty = resume, Ctrl+C = abort):`. Type a line
  and it's fed in as a steer exactly like `sessions send`; press Enter (or Ctrl+D) on an
  empty line to resume unchanged.
- **Second Ctrl+C** — either at that prompt, or before the checkpoint is reached —
  aborts, as it always has (the one-shot run exits 130 with its partial transcript
  saved; the REPL rolls the current turn back and keeps the session).

This is tty-only (a piped or `--json` run keeps the plain, non-interactive behavior)
and needs no flag or mailbox — it works even for a `--ephemeral` attached run. Under the
hood it's the same checkpoint as `sessions send`, so an in-flight tool call is never cut
off mid-write.

#### Personas (local system-prompt files)

Keep a library of your own reusable system prompts as plain `.md`/`.txt` files
under `~/.config/venice/personas/` — private, version-controllable, offline, and
complementary to Venice's server-side `--character` slugs. Drop a file in the
directory, then load it by bare name:

```sh
mkdir -p ~/.config/venice/personas
printf 'You are a terse pirate. Answer in one sentence.\n' \
  > ~/.config/venice/personas/pirate.md

venice chat --persona pirate            # load at launch
# ...or mid-session in the REPL:
#   /persona           -> list personas (name + first line)
#   /persona pirate    -> load personas/pirate.md as the system prompt
```

`/persona <name>` replaces the system prompt but keeps the conversation (use
`/reset` for a clean slate), exactly like `/system`. At launch, `--persona`
(or the `defaults.chat.persona` config default) seeds the system prompt; an
explicit `--system` / `defaults.chat.system` takes precedence. Names are
**bare only** — a name with a path separator or `..` is refused, and the lister
enumerates only the `personas/` directory, so a persona can never read the
neighbouring `credentials` file. `venice chat` only for now.

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
produces a final answer. These run **in-process on the `[openai]` extra alone**
(no `mcp` SDK, no subprocess):

`venice_image`, `venice_tts`, `venice_sfx`, `venice_music`, `venice_upscale`,
`venice_bg_remove`, and `venice_chat` (a sub-completion / subagent primitive) —
seven capabilities also exposed by `venice mcp-serve`. The MCP server additionally
adds `venice_video`, `venice_image_edit`, and its declared-host form of
`venice_vision`. The in-process agent set also includes `project_search`,
a read-only [semantic search](#semantic-search) over the project's local
`venice index` for locating code by meaning before acting on it (a **snapshot** of
the last index build — pair it with `reindex`, a paid tool that rebuilds the index
so recall reflects edits made this session), and
`venice_models`, a read-only lookup that lists model ids for a given catalog
type (its single `type` arg — text/code/image/video/music/tts/embedding/upscale,
or `all`) so the model can pick a valid `model` for the other tools instead of
guessing, and `venice_model_details` (single `model` arg) which returns one
model's pricing (cost), `capabilities` (text models — supportsVision etc.),
`constraints` (image/media — aspect ratios, resolutions, qualities, prompt-length
limit), and the full `model_spec`, so the agent can budget input and confirm a
model fits, and `venice_vision`, which accepts a local image (`input_path`, as a
base64 data-URL) or an `image_url` — the agent's eyes, so it can inspect its own
generations (watermarks, character consistency, glitches) instead of working
blind. `mode=auto` (the default) attaches the image to the active frontend when
that model advertises `supportsVision`; otherwise it delegates to a separate
vision model and returns that model's text. `mode=native` or `mode=delegate`
forces either path. An optional `prompt` directs the question; `model` and
`max_tokens` configure only delegation or the `auto` fallback. Not spend-gated.
(`venice code` gets all three too; `mcp-serve` uses the separate
[`--host-image-content`](#mcp-server) declaration for native vision.)

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
- **Session spend cap** (`--session-max-spend`, #66) meters the *chat completions
  themselves* — not just paid tools. Each turn's server-reported `usage` is
  priced against the session model's per-1M-token catalog rate; once the running
  total reaches the cap the loop stops starting new turns and forces a final
  answer (chat has no pre-call quote, so it bounds *further* spend, not a turn
  already in flight). What it caps is **starting new turns**, not every API call:
  auto-compaction's summarization calls are now counted toward the total, and are
  skipped when the cap has already tripped and no turn will follow — but a forced
  final answer still compacts if it must (shipping the full history would fail the
  turn outright), and an explicit `/compact` still runs. As of #117 it counts
  **every API call the CLI makes** — the main loop, auto-compaction, and all four
  subagent rails (`--scout`, `--spawn`, `--review`, `--web-search`). Each rail is
  priced against **its own** model's catalog rate, so a `--review-model` costlier
  than the coding model is billed as what it actually is, and each lands in its own
  off-loop bucket so the main loop's cache-hit signal stays uncontaminated (a
  subagent starts a fresh prefix every time, and averaging that in would fabricate a
  cache cliff). Config-backable via
  `defaults.chat.session_max_spend` /
  `defaults.code.session_max_spend`. Distinct from `--max-spend` (the per-call
  tool cap). A model with unknown pricing is counted (tokens) but not charged.
- `--output DIR` sets where generated files are written (default: cwd).
- **What the run cost.** A finished `--tools` run prints one line to **stderr** —
  `chat: 8.3s wall (5.1s tools) -- cost: $0.0041 (tokens prompt=3120 completion=284, cache 71.4% hit)`
  — so stdout
  stays exactly the deliverable and `venice chat --tools … | …` is unaffected. It
  totals *every* turn of the loop, not just the last one, and it prints on the
  Ctrl+C and API-error exits too, which are the runs most worth costing out. A run
  that never reaches a model call (an unknown `--tool`, a model that can't do
  function calling, a failed `--mcp` attach) prints nothing.
  Under `--json`, the stderr line is suppressed and stdout is an aggregate envelope:
  `{ "final": "...", "usage": {...}, "venice_parameters": {...} }`. Its `usage`
  is the same whole-run ledger as the human footer, including every model turn,
  wall/tool timing, cache accounting and off-loop spend. Before v0.83.36 this path
  printed the raw final completion object, whose `usage` covered only the last turn;
  the envelope is an intentional pre-1.0 contract break rather than retaining a
  plausible-looking undercount.
  Before v0.77 a default `--tools` run metered nothing at all: the ledger only
  existed when `--session-max-spend` was set.
- **Ctrl+C** during a `--tools` run prints `chat: aborted`, reports what the turns so
  far cost, and exits 130 (it used to raise a traceback).
- **Non-streamed in v1.** The tool path buffers each turn, so `--stream` is ignored
  when `--tools` is on; `--json` prints the aggregate envelope described above.

| flag | effect |
|---|---|
| `--tools` / `--agent` | enable the in-process tool-calling loop |
| `--tool NAME` | restrict to this tool (repeatable; default: all of them) |
| `--max-tool-calls N` | cap tool invocations before forcing an answer (default 8) |
| `--max-spend USD` | per-call auto-approve cap for paid tools |
| `--session-max-spend USD` | cap total chat-completion spend for the session |
| `--yes` / `-y` | auto-approve every paid tool call and side-effecting MCP tool |
| `--output DIR` / `-o` | directory for generated files |
| `--shell` / `--exec` | add a gated `shell` tool (`/bin/sh -c` in the cwd); implies `--tools` |
| `--shell-allow CMD` | allow only these commands for `--shell` (repeatable; adds to config `shell.allow`) |
| `--shell-deny PATTERN` | refuse commands matching these globs (repeatable; adds to config `shell.deny`) |
| `--shell-unrestricted` | acknowledge an empty allowlist under `--yes` (required for that combination) |
| `--browser` | add pinned `web_fetch` and sandboxed Chromium `browser_capture`; implies `--tools` |
| `--browser-allow HOST` | restrict destinations to matching host globs (repeatable; adds to `browser.allow`) |
| `--browser-deny PATTERN` | deny matching hosts or URLs; deny wins (repeatable; adds to `browser.deny`) |
| `--browser-private-host HOST` | authorize an exact private hostname (also requires a private range) |
| `--browser-private-range CIDR` | authorize a loopback/RFC1918/IPv6-ULA range (also requires an exact host) |
| `--memory` | add persistent memory + task tools (durable notes + a checklist); implies `--tools` (see [Memory & tasks](#memory--tasks---memory)) |
| `--mcp NAME` | attach a registered external MCP server's tools (repeatable) |
| `--no-mcp` | attach no MCP servers (overrides a configured default) |

#### Shell exec tool (`--shell`)

`--shell` (alias `--exec`) adds a gated **`shell`** tool so the agent can run any CLI
(`gh`, `git`, `curl`, build/test commands) — the same `/bin/sh -c` rail
[`venice code`](#coding-agent-venice-code) uses, with cwd set to the current directory,
a timeout, size-capped output, and the Venice API keys **scrubbed from the child
environment**. Every command is shown and **confirmed** before it runs (`[y/N]`), unless
`--yes` auto-approves. `--shell` implies `--tools`.

```sh
venice chat --shell --shell-allow gh --shell-allow git "Open my oldest assigned issue."
```

Scope it with an **allow/deny policy** — CLI flags add to a shared top-level `shell`
section in [config](#config) (`shell.allow` / `shell.deny`), the single source of truth
for both `venice chat --shell` and `venice code`'s `run` tool:

```sh
venice config set shell.deny  '["rm *", "sudo *", "* --force*"]'
venice config set shell.allow '["git", "gh", "ls", "cat"]'
```

- **Deny** globs are matched on the whole command line and on each token, are **always**
  enforced, and win over allow. Use `sudo` to block by name, `*rm -rf*` for a substring.
- A non-empty **allowlist** additionally requires a *single simple command* — no shell
  operators, pipes, redirects, substitutions, or variables (`; | & < > ( ) \` $`) — and
  the leading command's basename must be allowlisted (globs like `git*` are fine). This
  stops an allowlisted `gh && rm -rf ~` from slipping through.
- An **empty allowlist is unrestricted** (only the confirm gate + deny apply). Combining
  that with `--yes` (auto-approved arbitrary shell) is refused unless you pass
  `--shell-unrestricted` to acknowledge it.
- **Not exposed over `venice mcp-serve`** — a shared/remote server running arbitrary
  shell is a much larger blast radius and is deliberately out of scope.

#### Web & browser tools (`--browser`)

`--browser` opts into two read-only tools. `web_fetch` retrieves bounded HTTP(S) text or
HTML. `browser_capture` renders post-JS DOM/text and/or a PNG with a Chromium-family
browser. The browser runs non-root with its normal sandbox enabled, a disposable profile
and home,
an allowlisted environment, and no ambient credentials. Firefox is not used because its
CLI cannot provide the same observable, proxy-enforced capture contract.

The network boundary is enforced at connection time. Each destination is resolved once,
the complete answer set is checked, and the socket connects only to one of those numeric
addresses. HTTPS retains the original hostname for SNI, certificate verification, and
`Host`. Redirects are capped at five and rechecked. Chromium receives one loopback policy
proxy for HTTP, HTTPS, WebSockets, redirects, frames, scripts, styles, images, fonts,
fetch/XHR, workers, service workers, prefetches, and downloads; there is no direct fallback.
QUIC and non-proxied WebRTC UDP are disabled. A capture is capped at 60 seconds, 128
proxy connections, 16 concurrent connections, and 32 MiB total proxied traffic; `web_fetch`
is capped at 2 MiB and 60 seconds.

Public Internet addresses are allowed by default unless narrowed with `browser.allow` or
denied with `browser.deny` (deny always wins). Loopback and private networks are denied by
default. Private access needs two independent operator grants: the exact canonical host in
`browser.private_hosts` and every resolved address inside `browser.private_ranges`.
Ranges may only be loopback, RFC1918, or IPv6 ULA. Link-local, metadata, multicast,
reserved, unspecified, and IPv4-mapped IPv6 destinations remain hard-blocked and cannot be
reopened. The model schemas contain none of these authority fields.

```sh
venice chat --browser --browser-allow docs.example.com "Read the installation page"

# Deliberate local development access: both grants are required.
venice code --browser \
  --browser-private-host 127.0.0.1 \
  --browser-private-range 127.0.0.1/32 \
  "Check the app at http://127.0.0.1:3000"

venice config set browser.private_hosts '["dev.internal"]'
venice config set browser.private_ranges '["10.20.0.0/16"]'
```

Threat model: pages, redirects, DNS answers, and model-supplied URLs are untrusted. The
operator controls reachability; a model may use only the already-bound policy. This is a
network and process-containment boundary, not a promise that page content is trustworthy.
Do not run `browser_capture` as root, disable the Chromium sandbox, or reuse its profile.

#### Web search (`--web-search`)

`--web-search` adds one discovery tool, `venice_web_search`, so the coding
agent can look something up on the web when a fix needs documentation it doesn't have a link
for (an API's docs, a library's usage, an error message).

`venice_web_search(query)` makes **one** Venice web-search completion (server-side
`enable_web_search` + `enable_web_citations`, the same feature behind `venice chat
--web-search`) and returns a short **answer** plus the **cited URLs**. To then read a cited
page in full, opt into `--browser` and use the pinned `web_fetch` follow-up rail.

```sh
# discover: search returns an answer and its cited URLs
venice code --web-search --auto \
  "The stripe SDK call is failing with an idempotency error — look up the fix and apply it."
```

- **Model.** Web search needs a model advertising `supportsWebSearch`. By default the
  coding `--model` is used when it qualifies, else the first capable model in the catalog;
  override with `--web-search-model MODEL` (or `defaults.code.web_search_model`). No model id
  is hard-coded — it's resolved against the live `/models` catalog. The resolved id and
  whether it came from the flag, config, or automatic selection are printed before the
  first paid call; unknown ids and models known to lack web search fail at startup.
- **Billed, bounded.** It rides the normal completion path (same key, same billing) rather
  than a scraper, so there's no new dependency or secret. It isn't per-call spend-gated; its
  cost is bounded by the agent's tool-call budget, and each result carries a best-effort
  `cost_estimate_usd`.
- **Who gets it.** The coding agent (and a `--planner`) can use it directly; with `--scout`
  the read-only scout becomes a **"docs scout"** (read the tree *and* search the web). Spawn
  **workers never get it** — a worker acting on instructions injected via a search result is
  the blast-radius case, so web tools are denied to workers by default.
- Config: `defaults.code.web_search` (bool) / `defaults.code.web_search_model`. Not exposed
  over `venice mcp-serve`.

#### Memory & tasks (`--memory`)

`--memory` gives the agent a durable place to keep notes and a checklist, so multi-step
and cross-session work survives beyond one transcript. Implies `--tools`. Seven free,
local, offline tools (no API calls, no spend gate):

- **`memory_write` / `memory_read` / `memory_search` / `memory_list`** — named notes the
  agent recalls later. **Two tiers:** `scope="project"` (default) rides the repo at
  `<root>/.venice/memory/` so it's shared by anything working in that tree; `scope="global"`
  lives user-global (travels with the agent across projects). `memory_search` is a plain
  substring match over names/descriptions/bodies (zero-dep, always works); `memory_list`
  returns metadata only (names/types/descriptions/timestamps — a cheap index to decide what
  to read).
- **`task_add` / `task_update` / `task_list`** — a lightweight **project-only** checklist
  (`pending` → `in_progress` → `done`) the agent maintains across turns and `--resume`.

```sh
venice code --memory --auto "Refactor the parser; track your steps as tasks."
venice chat --memory "Remember that this project uses tabs, not spaces (scope=project)."
```

Inspect or prune what the agent stored with the **`venice memory`** command:

```sh
venice memory ls                 # both tiers (metadata only)
venice memory ls --scope global  # just the global tier
venice memory show <name>        # one note, including its body
venice memory rm <name>          # delete a note (default: project tier)
venice memory tasks              # the project checklist (--status filters)
```

- **Locations:** project notes/tasks at `<root>/.venice/memory/` (git-ignored by default,
  like the semantic index); global notes at `~/.config/venice/memory/`
  (`$VENICE_MEMORY_DIR` overrides). All store files are `0600`.
- **Hygiene (AGENTS.md):** a note **name** is refused if it's secret-shaped
  (`credentials`, `id_rsa*`, `*.key`, `.env`, `*secrets*`, …), so the store can't be used
  to label or stash a credential.
- **Not exposed over `venice mcp-serve`** (chat/code only), like the browser rails.

#### External MCP tools (`--mcp`)

`--mcp NAME` attaches the tools of an external [MCP](https://modelcontextprotocol.io)
server (filesystem, git, shell, ...) **alongside** the built-in venice tools, so one
agent can drive both. Register servers first with
[`venice config add`](#config) (stdio or http/sse), then name them:

```sh
venice config add fs --command npx --arg -y --arg @modelcontextprotocol/server-filesystem --arg /work
venice chat --mcp fs "Summarize the TODOs across the source files."
```

- **Needs the `[mcp]` extra** (`pip install "venice-cli[mcp]"`, Python ≥ 3.10).
  `--mcp` implies the agent loop (no separate `--tools` needed); it still requires a
  function-calling model and degrades to plain chat otherwise.
- Remote tools are advertised as `server__tool` (namespaced to avoid collisions).
- **Side-effecting tools are gated.** A remote tool that isn't annotated read-only
  prompts for confirmation on a TTY (or feeds the request back to the model
  non-interactively) before it runs; read-only tools run freely. `--yes` bypasses
  the gate. This rides the same confirm rail as paid built-in tools.
- Multiple `--mcp` flags attach multiple servers; `--no-mcp` overrides a
  `defaults.chat.mcp` config default. Attach timeouts: `$VENICE_MCP_CONNECT_TIMEOUT`,
  `$VENICE_MCP_CALL_TIMEOUT`.
- Works the same in [interactive mode](#interactive-mode) — servers stay attached for
  the whole session and are torn down on exit.

## Embeddings

Turn text into embedding vectors with a Venice embedding model (`/embeddings`,
via the OpenAI SDK). Venice advertises no `default`-trait embedding model, so
`--model` is required (see `venice models --type embedding`); set
`defaults.embed.model` to stop repeating it. The CLI refuses to pick one for you
rather than silently charging for an arbitrary choice — embeddings are only
comparable within one model's vector space, so a silent default would emit
vectors from a different space into an existing store. With neither a `--model`
nor a config default, `venice embed` exits 6 and names the config key.

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

# Or set the model once and stop passing --model.
venice config set defaults.embed.model text-embedding-bge-m3

# Full raw response object (model, data, usage) instead of bare vectors.
venice embed "hello" --json | jq '.usage'
```

By default each embedding prints as a JSON array, one per line;
`--encoding-format base64` requests base64-packed vectors instead of floats.

### Local / alternate backend

Because `venice embed` rides on the OpenAI SDK, it can point at **any**
OpenAI-compatible embeddings endpoint — including a local one (llama.cpp,
Ollama, text-embeddings-inference). Pass `--embed-base-url` (with
`--embed-model`, since the alternate server has its own catalog) to swap
backends; this skips the Venice catalog and needs no Venice key. Venice stays
the default when the flag is absent.

```sh
# Embed against a local server -- no Venice key required.
venice embed --embed-base-url http://localhost:1234/v1 \
    --embed-model my-local-model "the quick brown fox"
```

The URL can also come from `$VENICE_EMBED_BASE_URL`, and a key (if the backend
needs one) from `$VENICE_EMBED_API_KEY` or the **named-secret store** (see
[`venice secret`](#secrets-venice-secret)) — never `config.json`. Persist it once
with `venice login --embed` (or `venice secret set embed`) to retire the
`export VENICE_EMBED_API_KEY=…` in your shell profile; the env var still wins when
set. The base-url/CA-bundle flags are config-backable per-flag via `defaults.embed.*`
(see `venice config`).

**Self-signed backends.** A local embedder fronted by Traefik/Caddy often serves
a private or self-signed TLS cert, which the OpenAI SDK rejects
(`embed: connection error`, exit 8). Two opt-in escape hatches — **applied only
to `--embed-base-url`, never the Venice endpoint**:

```sh
# Trust a private CA (verification stays ON -- preferred):
venice embed --embed-base-url https://embed.local/v1 --embed-model bge-m3 \
    --embed-ca-bundle /etc/ssl/my-ca.pem "hi"          # or $VENICE_EMBED_CA_BUNDLE

# Disable verification entirely (self-signed, no CA handy -- prints a warning):
venice embed --embed-base-url https://embed.local/v1 --embed-model bge-m3 \
    --embed-insecure "hi"
```

`--embed-ca-bundle` is config-backable (`defaults.embed.embed_ca_bundle`) and
reads `$VENICE_EMBED_CA_BUNDLE`. `--embed-insecure` is CLI-only by design —
turning verification off should always be an explicit, visible choice, never
something a stale env var or config file switches on. The two are mutually
exclusive, and passing either without `--embed-base-url` is an error (exit 2).

## Semantic search

`venice index` builds a local semantic index of a project tree, and `venice
search` finds the chunks most relevant to a natural-language query by meaning
rather than by keyword. Both use the same embedding machinery as `venice embed`
(the `[openai]` extra, Venice **or** a local backend); the vector store and the
cosine search are pure-stdlib, so no extra dependency is needed.

```sh
# Index the current tree. Venice has no default embedding model, so pass one
# (or set defaults.index.model). Vectors land in ./.venice/index/.
venice index . --model text-embedding-bge-m3

# Search it from anywhere in the tree (walks up to find .venice/index).
venice search "where is the retry/backoff logic"

# Top-3 results as JSON (path, line range, score, preview).
venice search "jwt refresh handling" -k 3 --json
```

Text output is one hit per line as `SCORE  path:start-end`, followed by a short
preview of the matched lines:

```
0.8137  src/venice/client.py:88-120
    def post_for_bytes_or_json(self, path, body, ...):
```

**Incremental.** Re-running `venice index` re-embeds only the files whose
contents changed (keyed on a SHA-256 of each file); unchanged files keep their
vectors and deleted files are dropped. `--rebuild` forces a full re-index — also
required if you switch model/dimensions/backend, since vectors from different
embedding spaces are not comparable.

**What gets indexed.** UTF-8 text files under the tree, chunked into overlapping
line windows (`--chunk-lines` / `--chunk-overlap`). The walker skips binaries,
oversized files, `.git`/`node_modules`/virtualenvs and similar, and honors a
simple top-level `.gitignore` plus any `--exclude GLOB`. Credential- and
secret-shaped files (`.env`, `credentials`, `*.pem`, `*.key`, `id_rsa*`) are
**never** indexed, and symlinks pointing outside the tree are ignored.

**Local backend.** As with `venice embed`, `--embed-base-url` (+ `--embed-model`,
or `$VENICE_EMBED_BASE_URL` / `$VENICE_EMBED_API_KEY`) points indexing at a local
OpenAI-compatible server — cheap for embedding a whole tree, and needs no Venice
key:

```sh
venice index . --embed-base-url http://localhost:1234/v1 --embed-model bge-m3
venice search "parse the queue response"   # uses the index's own backend/model
```

**Self-signed backends.** Both commands accept the same TLS escape hatches as
`venice embed` — **applied only to a local backend, never the Venice endpoint**:

```sh
venice index . --embed-base-url https://embed.local/v1 --embed-model bge-m3 \
    --embed-ca-bundle /etc/ssl/my-ca.pem        # trust a private CA (or $VENICE_EMBED_CA_BUNDLE)
venice search "parse the queue response" --embed-ca-bundle /etc/ssl/my-ca.pem
venice index . --embed-base-url https://embed.local/v1 --embed-model bge-m3 \
    --embed-insecure                            # disable verification (warns; CLI-only)
```

`--embed-ca-bundle` reads `$VENICE_EMBED_CA_BUNDLE` and is config-backable
(`defaults.index.embed_ca_bundle` / `defaults.search.embed_ca_bundle`);
`--embed-insecure` is CLI-only, mutually exclusive with it, and errors (exit 2) if
the flags don't apply (no `--embed-base-url` for `index`, or a Venice-built index
for `search`). For `search` the CA bundle is supplied fresh at query time — it is
never baked into the index — and the `project_search` agent tool also honours
`$VENICE_EMBED_CA_BUNDLE`, so a `venice chat`/`venice code` session can search an
index built against a self-signed embedder.

The index is machine-generated: `venice index` drops a self-ignoring
`.venice/.gitignore`, so it won't be committed even if your repo doesn't already
ignore `.venice/`. Config-backable per-flag via `defaults.index.*` /
`defaults.search.*`. `venice search` is also exposed to the chat agent as the
`project_search` tool (see **Agent / tool calling**), so a `venice chat --tools`
session can locate code by meaning before acting on it.

> **`project_search` is a snapshot; `grep` is live.** The `.venice` index is a
> point-in-time build, so `project_search` returns pre-edit content for files the
> agent changed this session, while the `grep` tool always walks the working tree.
> After edits, the agent can call the **`reindex`** tool to rebuild the index (it
> re-embeds only the files whose contents changed, reusing the index's existing
> embedding backend). `reindex` is **paid** (it calls the embeddings API) and
> routes through the same confirm gate as the other paid tools — approve it at the
> `y/a/N` prompt (or run non-interactively with `--yes`). If no index exists yet it
> tells you to run `venice index` first.

## Coding agent (venice code)

`venice code` is a self-contained coding agent ("vcoder") built on the tool loop.
Point it at a project and give it a task: it **proposes a plan, waits for your
acceptance, then reads, edits, and runs commands** using built-in, path-sandboxed
tools, powered by a function-calling Venice model. Needs the `[openai]` extra and a
tool-calling model (unlike `venice chat --tools`, it errors out rather than degrading
if the model can't call tools). The coding engine itself is pure stdlib — no new
dependency.

```sh
# Human at a terminal: see the plan, then choose auto/step at the prompt.
venice code "add retry with backoff to the HTTP client and a test" -m mistral-31-24b

# Autonomous, unattended (a script/cron): accept + run to completion, JSON out.
venice code --auto --json "bump the version and update CHANGELOG" > result.json

# Two-step (for a script or a controlling LLM to approve out of band):
venice code --plan-only --json "refactor the parser" > plan.json   # prints plan, exits
venice code --auto "refactor the parser"                            # then execute

# An interactive coding session (tools on; changes confirm per step).
venice code -i
```

**Plan → acceptance → run.** The command always plans first (one no-tools turn that
emits a numbered plan + acceptance criteria), then crosses an **acceptance boundary**
three possible ways, then executes and finally self-checks the criteria:

| How it's launched | How the plan is accepted | Run mode |
| --- | --- | --- |
| Human, terminal | Interactive prompt: `[a]uto / [s]tep / [e]dit / [N]o` | chosen at the prompt |
| Flag-driven | `--auto` (accept + autonomous) or `--manual` (accept + step) | from the flag |
| Out of band | `--plan-only` prints the plan and exits 0; the caller re-invokes to run | deferred |

Non-interactive with neither `--auto` nor `--plan-only` **aborts (exit 2)** before any
model call — side-effecting work never runs unattended without an explicit opt-in.
After execution a final turn reports each criterion MET/NOT MET and ends with an
`ACCEPTANCE: PASS`/`FAIL` verdict; with `--json` the verdict lands in the envelope
(`acceptance.verdict` = `pass`/`fail`/`unknown`). The verdict parse is
case/format-tolerant and **re-prompts once** for the verdict line if the first reply
lacks it, so a correct run whose model phrased its verdict loosely still exits 0. The
**exit code reflects it**: 0 = all met (or check skipped), 1 = not met, 10 = the model
never emitted a parseable verdict even after the re-prompt (the work may still be
complete — a loud stderr warning is printed).

**What the run cost.** A finished run prints one line to **stderr** —
`code: 2m 14s wall (1m 02s tools) -- cost: $0.0431 (tokens prompt=48201 completion=3904, cache 94.0% hit)`
— so
stdout stays exactly the deliverable and `venice code … | …` is unaffected. It
prints on the Ctrl+C and API-error exits too, which are the runs most worth
costing out. Under `--json` the line is suppressed and the numbers ride the
envelope's `usage` key instead, in the same shape the session file stores, so
`venice code --json | jq .usage` and `jq .usage <session>.json` agree. The total
covers the plan turn and the acceptance turns as well as the tool loop; the
wall-clock excludes time spent at the plan-accept prompt. Note that runs now
always carry a `usage` blob in their session file — previously only
`--session-max-spend` runs metered at all.

**Watching the cache.** A prompt-cache collapse is a silent 3-5× cost event, so the
active model's current-run hit rate rides that same footer (and `/cost`), not just
`/usage`. It reports `cache n/a` when the model's usage block carried no cache fields
at all — a printed percentage always means the provider measured something. The
machine-readable model dimensions are `usage.models` (cumulative session usage) and
`usage.current_run_models` (only this process/resume leg), both keyed by exact model
id. Each row carries `cache_hit_percent`, token/cost counters, reporting flags, and
the main-loop API-call count:

```sh
venice code --json "..." | jq '.usage.current_run_models'
jq '.usage.models' ~/.config/venice/sessions/<id>.json
```

The legacy top-level counters and `usage.cache_hit_percent` remain cumulative for
JSON compatibility and as the `--session-max-spend` authority; do not use that
cross-model percentage for cache forensics. Alert on a row under
`current_run_models` and gate on that row's `cache_read_unreported` being `false`.
A partially reported row is marked `[partially unreported]` in human output.
`VENICE_USAGE_RAW=1` dumps each response's raw `usage` block to stderr when you need
to see which turns those were.

**Where the time went.** "The model is slow" and "my test suite is slow" call for
completely different responses, so the wall figure carries a tool-time clause —
`2m 14s wall (1m 02s tools)` — and `/usage` itemizes it:

```
  wall        2m 14s  over 1 turn(s)  (avg 2m 14s)
  tools       1m 02s  across 19 call(s)
    shell             48.1s   7 call(s)
    apply_patch       11.4s   5 call(s)
    read_file          2.5s   7 call(s)
```

Only `tool.invoke` is counted, so the seconds you spend at a `Proceed?` confirm
prompt are *not* charged to the tool. The tools total is a **subset** of wall — the
model wait, auto-compaction and the plan/acceptance turns are all wall time no tool
owns — with one exception: under `--parallel`, overlapping subagent windows both land
in the total, so it can exceed wall and is then marked `concurrent`. Long runs fold
the tail into a `(+N more)` row that carries its own seconds and calls, so the rows
always add back up to the header. The machine-readable half is `usage.tool_seconds`
and `usage.tools`, in both the `--json` envelope and the session file:

```sh
venice code --json "..." | jq '.usage.tools.shell.seconds'
venice code --json "..." | jq '.usage.tool_seconds'
```

**Why the cache rate went bad.** A low session-wide hit rate has at least three very
different causes that all average to the same number: the provider never cached at
all, the prefix is churning, or auto-compaction is re-firing and resetting it each
time. The aggregate cannot tell them apart, so `/usage` also traces the individual
API calls:

```
  calls        39.7s  across 14 API call(s)  [+1 off-loop]
    #1              11,204 in    0% cached      312 out      4.2s
    #2              11,530 in   96% cached       88 out      1.1s
    -- compacted (auto) after #2: 48 -> 13 msgs, ~91,200 -> ~8,210 tok est, $0.0031 to summarize
       (88,110 tok measured before, lower bound)
    #3               8,402 in    0% cached      140 out      2.0s
    (+6 elided)     76,500 in   58% cached      780 out     15.0s
    #10             16,600 in   58% cached      158 out      3.2s
    #11             17,700 in   58% cached      166 out      3.4s
    #12             18,800 in   58% cached      174 out      3.6s
    #13             19,900 in   58% cached      182 out      3.8s
    #14             19,880 in   91% cached      210 out      3.4s
  off-loop     36.1s  across 9 API call(s)  [not in the trace above]
    compaction  1 call(s)     88,110 in      420 out     $0.0031
    review      3 call(s)     12,000 in    1,800 out     $0.0090
    scout       4 call(s)      6,200 in      900 out     $0.0038
    web_search  1 call(s)        800 in      120 out     $0.0008
```

Cold-from-#1, a mid-run cliff and a compaction sawtooth each have a distinct shape
here. `n/a` means *unknown* and never zero — a response that carried no cache block
reads `n/a cached`, and an API call whose window was not stamped reads `n/a` for its
seconds, exactly as the session-wide rate refuses to print a fabricated `0.0%`.
Compaction markers are never elided, and the elided span carries its own totals so
the rows still add back up to the header.

The `off-loop` block is the calls this CLI made that were **not** conversation turns:
compaction's own summarization call, plus one row per subagent rail (`scout`, `spawn`,
`review`, `web_search`). They are billed and shown, but kept out of the trace above on
purpose: each one is a fresh prefix that reads ~0% cached, so averaging it in would
manufacture exactly the cliff the trace exists to detect. `[+1 off-loop]` on the header
is what stops the smaller `across 14` reading as the whole story.

Each rail is priced against **its own** model — `--review-model` is deliberately a
different model from the one authoring, and billing its tokens at the author's rate
would be a fabricated number. Rail seconds are a *subset* of the wall clock and are not
disjoint from the `tools` block: a rail's API time sits inside the `venice_scout` /
`venice_spawn` / `venice_review` tool window containing it, and under `--parallel`
several rails overlap each other, so the block says `[concurrent -- exceeds wall]`
rather than quietly summing past the wall clock.

Note that **one API call is not one turn**: `usage.turns` counts the times the CLI
made you wait (one whole `code` run, one REPL turn), while a single turn that
dispatches tools makes several API calls. The machine-readable half is
`usage.api_calls` / `usage.api_calls_total` / `usage.context_events`, in both the
`--json` envelope and the session file:

```sh
# every call's hit rate, in order -- null means the provider reported nothing
venice code --json "..." | jq '.usage.api_calls[] | {n, model, prompt_tokens, cache_read_tokens}'
# did compaction fire, and what did it cost the prefix?
jq '.usage.context_events' ~/.config/venice/sessions/<id>.json
```

Rows are capped at the first 50 plus the most recent 200, so a long session keeps
both its cold-start evidence and its current state; each row carries its `n`, so a
jump in the sequence is itself the drop marker. Sessions saved before model-aware
accounting are attributed to their saved session model when one is available; an
import with no model provenance retains its cumulative totals without inventing a
model bucket.

`venice code` also watches that trace for the narrow total-collapse case. Once the
selected model advertises `pricing.cache_input`, API call 2 or later has at least
2,000 prompt tokens, and the provider explicitly reports **zero** cached tokens, the
default `--cache-guard warn` prints one stderr diagnostic for that run. An absent
cache field is unknown and never trips the guard. `--cache-guard stop` completes any
tool results required by the response that exposed the collapse, then requests one
final answer with tools disabled; if the response was already final it buys no extra
call. `off` disables the policy. Compaction and the fresh-prefix off-loop rails stay
outside the guard for the same reason they stay outside the trace.

Compaction's summarization call and every subagent rail **are** metered, each into its
own `usage.buckets` partition rather than into the rows above — they are fresh prefixes,
so they read ~0% cached every time and would fabricate a cache cliff if averaged in with
the conversation. The event row carries what one compaction cost; the bucket carries the
running total. `usage.api_calls_total` therefore counts **main-loop** calls only, and the
`calls` header names the difference (`[+2 off-loop]`).

The run footer names the off-loop share too, but stays short: one or two buckets are
listed (`[incl. $0.0031 compaction]`), and past that it collapses to
`[incl. $0.0180 off-loop]` and defers to `/usage` for the per-rail breakdown.

To distinguish "the cache stopped working" from "the provider stopped reporting
it," run a controlled live probe with synthetic content rather than project or user
data:

```bash
# Configured chat model plus a cache-advertising model from another family.
venice cache-probe

# Explicit bounded threshold sweep; six-digit prefixes are never a default.
venice cache-probe --model kimi-k3 \
  --prefix-tokens 8192 --prefix-tokens 64000 \
  --prefix-tokens 112000 --prefix-tokens 120000 --prefix-tokens 128000

# Machine-readable result, with a hard cap for the complete matrix.
venice cache-probe --model kimi-k3 --repeat 3 --json --yes --max-spend 0.50
```

The command prints a conservative all-uncached estimate before making any paid
completion call, then requires confirmation (or `--yes`). `--max-spend` fails
closed if the catalog price is absent or the complete matrix exceeds the cap.
Each call reports the raw `prompt_tokens_details` object unchanged; each model/size
is classified as `warms`, `never warms`, or `field absent`. Repeating
`--prefix-tokens` makes warming below a threshold and failure above it visible,
while size-specific prefixes prevent one sweep row from warming another.

```bash
# what has compaction cost this session?
jq '.usage.buckets.compaction' ~/.config/venice/sessions/<id>.json
# what did the rails cost -- which one is eating the budget?
jq '.usage.buckets' ~/.config/venice/sessions/<id>.json
# the whole bill, main loop + off-loop
jq '.usage.billed_total' ~/.config/venice/sessions/<id>.json
```

**Tools** (path-sandboxed to the project root; mutating tools confirm unless `--auto`):

| Tool | Does | Confirms? |
| --- | --- | --- |
| `read_file` / `list_dir` / `grep` | read a file, list a dir, regex-search the tree | no |
| `git` | validated read-only Git (`status`/`diff`/`log`/`show`/…); literal paths go after `--` | no |
| `project_search` | semantic search over the `.venice` index (if built) — a **snapshot** of the last build; use `grep` for live matches | no |
| `reindex` | rebuild the `.venice` index so `project_search` reflects this session's edits (re-embeds only changed files); present only when an index exists | yes (paid) |
| `venice_vision` | inspect a local image or URL natively on a capable frontend, otherwise through a delegated vision model (`mode=auto\|native\|delegate`) | no |
| `write_file` | create/overwrite a file (atomic) | yes |
| `edit_file` | replace an exact, unique string in a file | yes |
| `apply_patch` | apply a batch of edits grouped per file, atomically per file (use `occurrence=N` for non-unique strings) | yes |
| `run` | run a shell command (`/bin/sh -c`) at the **active** root | yes |
| `attach_root` | register another directory as a project root (for work spanning repos) and, by default, switch the active root into it so relative paths and `run`/`git` follow — writes outside the writable roots fail loudly | no |
| `venice_image` / `venice_image_edit` / `venice_sfx` / `venice_music` / `venice_tts` / `venice_upscale` / `venice_bg_remove` / `venice_video` | generate/edit images, audio & video into the project — **opt-in with `--assets`** | yes |
| `web_fetch` / `browser_capture` | bounded pinned fetch / sandboxed Chromium render — **opt-in with `--browser`** (see [Web & browser tools](#web--browser-tools---browser)) | no |
| `memory_write` / `memory_read` / `memory_search` / `memory_list` | durable notes the agent recalls across turns/sessions (two tiers: project + global) — **opt-in with `--memory`** (see [Memory & tasks](#memory--tasks---memory)) | no |
| `task_add` / `task_update` / `task_list` | a project-only checklist the agent tracks (`pending`/`in_progress`/`done`) — **opt-in with `--memory`** | no |
| `venice_scout` | delegate a read-only investigation to a disposable subagent with a fresh context; returns a structured report so exploration doesn't pollute the main context — **opt-in with `--scout`** (see [Scout subagent](#scout-subagent---scout)) | no |
| `venice_web_search` | search the web to **discover** documentation you don't have a URL for; returns a short answer + cited URLs (billed, but bounded by the tool-call budget) — **opt-in with `--web-search`** (see [Web search](#web-search---web-search)) | no |
| `venice_review` | hand the current diff to a **cold-context reviewer** — a disposable subagent with a fresh context, read-only tools and (where the catalog allows) a different model — and get back defects with `file:line` + a repro. Findings only: it approves and certifies nothing — **opt-in with `--review`** (see [Reviewer rail](#reviewer-rail---review)) | no |
| `venice_context_archive` | list bounded metadata or page through exact messages removed by evidence-preserving compaction; current session only, with no filesystem, network, or process access | no |

**Safety.** Every filesystem path is resolved and confined to the **writable roots** —
the startup root (default: cwd, or `--root` / `$VENICE_CODE_ROOT`) plus any added with
`--allow-root` / config `roots.allow` / the `attach_root` tool. A **write** that lands
outside the writable set **fails loudly** (naming the roots) rather than silently
redirecting — so a session that spans repos can't leak files into the wrong one; deny
roots (`--deny-root` / `roots.deny`) are readable but never writable (deny wins). This
is a guardrail, not a sandbox: a *shell command* can still write anywhere. A path that
also names a secret-shaped file (`credentials`, `.env`, `*.pem`, `*.key`, …) or lives
under `.git`/`.venice` is refused — the same denylist `venice index` uses. `run` executes
with the working directory forced to the **active** root, a timeout (`--exec-timeout`),
size-capped output, and the Venice API keys scrubbed from the child environment. Note
that a *shell command* can still touch paths outside the root (`cat ../x`); `run`'s
boundary is the **confirm gate** (the exact command is shown before it runs) plus the
forced cwd, timeout, and env-scrub — which is why it always confirms. git mutations
(`add`/`commit`) go through the gated `run` tool. `run` also honors the shared
[`shell` allow/deny policy](#shell-exec-tool---shell) (config `shell.*` or
`--shell-allow`/`--shell-deny`) — a denied command is refused; an empty policy leaves
it unrestricted (unchanged behavior).

The same active/attached readable-root and secret-path policy applies when
`venice_vision` or a `--assets` media tool reads a model-supplied local file. Those
inputs must also match a recognized media signature; confirmation and `--auto` do
not widen this authority. Native images remain in the conversation history so the
frontend can inspect them again; a persistent session therefore stores the bounded
data URL in its existing 0600 envelope.

The free `git` tool validates arguments per operation rather than trusting a
subcommand name. `branch` and `remote` are listing-only; content diffs require
literal, root-confined paths after `--`; pathspec magic, `REV:path`, `--no-index`,
output files, external diff/text conversion, and config injection are refused before
Git starts. Path operands must resolve to existing regular files, so a directory cannot
smuggle protected descendants into a broad diff. Git also runs without inherited
`GIT_*` controls, optional locks, system/global config, hooks, fsmonitor, pagers, or
external diff helpers. Use the confirmed `run` tool for other Git forms.

| flag | effect |
| --- | --- |
| `--auto`, `-y` | accept the plan and run autonomously (auto-approve every tool call); required to run with no terminal |
| `--manual` | accept and run with per-step confirmation (default on a terminal) |
| `--plan-only` | print the plan and exit without executing |
| `--no-plan` | skip the planning turn and execute directly |
| `--no-verify` | skip the post-run acceptance-criteria check |
| `--root DIR` | project directory to sandbox to (default: cwd) |
| `--allow-root DIR` / `--deny-root DIR` | extra directories the file tools may read+write / roots excluded from writes (repeatable; adds to config `roots.*`). The agent can also add roots at runtime with `attach_root` |
| `--max-tool-calls N` | cap tool invocations before forcing a final answer (default 25) |
| `--exec-timeout SECS` | timeout for `run`/`git` (default 120) |
| `--shell-allow CMD` / `--shell-deny PATTERN` | scope the `run` tool with the shared allow/deny policy (repeatable; adds to config `shell.*`) |
| `--browser` / browser policy flags | expose pinned fetch and sandboxed Chromium tools; public by default, allow/deny globs optional, private access requires both `--browser-private-host` and `--browser-private-range` (see [Web & browser tools](#web--browser-tools---browser)) |
| `--assets` | also expose the in-process asset-generation tools (image / image-edit / sfx / music / tts / upscale / bg-remove / video) so the agent can create images, audio & video in the project; paid — each confirms per call unless `--auto` |
| `--scout` | expose `venice_scout`: delegate a read-only investigation to a disposable subagent with a fresh context; keeps exploration out of the main context (see [Scout subagent](#scout-subagent---scout)) |
| `--spawn` | expose `venice_spawn`: delegate a bounded **write/paid** task to a disposable **worker** subagent with a fresh context and a role-scoped subset of your tools; edit churn stays quarantined and it returns a structured report to merge (see [Worker subagent](#worker-subagent---spawn)) |
| `--spawn-max-spend USD` | per-worker USD cap on an `asset` worker's cumulative estimated media spend (default **$2.00**; `<= 0` disables); config `defaults.code.spawn_max_spend` |
| `--subagent-max-tokens N` | per-subagent cap on the cumulative prompt+completion **tokens** a `venice_scout` **or** `venice_spawn` subagent spends across its turns (default **off**; `<= 0` disables); once crossed the subagent is asked to wrap up and its report carries the token count. A cumulative-usage ceiling, **not** a context-window size limit, and distinct from `--max-tokens` (per-turn output); config `defaults.code.subagent_max_tokens` |
| `--planner` | planner harness: implies `--scout --spawn --memory`, mandates the decompose → dispatch → track → **merge** protocol, and adds `venice_merge` — a consolidated rollup of every dispatch (see [Planner harness](#planner-harness---planner)) |
| `--parallel` | dispatch **independent** `venice_scout`/`venice_spawn` subagents **concurrently** (bounded pool) instead of one at a time, so a planner's independent units overlap in wall-clock; opt-in, serial otherwise; config `defaults.code.parallel` (see [Parallel dispatch](#parallel-dispatch---parallel)) |
| `--web-search` / `--web-search-model MODEL` | expose `venice_web_search` so the agent can **discover** documentation on the web (answer + cited URLs; see [Web search](#web-search---web-search)) |
| `--review` | expose `venice_review`: hand the current diff to a **cold-context reviewer** (fresh context, read-only tools, a different model where one exists) and fix what it reports before handing work back. Capped at **3 reviews per session**. Findings only — a review is not a merge gate (see [Reviewer rail](#reviewer-rail---review)) |
| `--review-model MODEL` | model for `--review` (default: a function-calling model from a **different family** than the coding model; falls back to the coding model with a warning); config `defaults.code.review_model` |
| `--review-rounds N` | passes `venice_review` makes over the same diff (default **1**, max 3); config `defaults.code.review_rounds` |
| `--auto-compact` | summarize older history once the prompt crosses `--compact-threshold` tokens (default 100 000), keeping the last `--compact-keep-turns` turns (default 10); long runs stay in-context |
| `--compact-loss-policy aggressive\|evidence` | discard summarized messages or retain their exact JSON in the bounded private session archive; default **evidence** for code and **aggressive** for chat; config `defaults.code.compact_loss_policy` / `defaults.chat.compact_loss_policy` |
| `--cache-guard off\|warn\|stop` | react when a cache-priced model explicitly reports zero cached tokens after the cold first API call and a 2,000-token prompt; default **warn**, config `defaults.code.cache_guard` |
| `-i`, `--json`, `--model`, `--system` | interactive REPL · JSON envelope · model · extra system instructions |
| `--persona NAME` | load `~/.config/venice/personas/NAME.md` as the system prompt at launch (`/persona` in the REPL) |

When review or web search is enabled, `--json` includes a `resolved_models` object whose
`review` / `web_search` rows carry the resolved `id`, a `source` of `flag`, `config`, or
`auto`, and the exact `config_key` for config-sourced values. Saved session envelopes carry
the same object. Disabled rails have no row, and invalid auxiliary models fail before the
planning or agent completion.

With `--assets`, generated files land in `$VENICE_MCP_OUTPUT_DIR` or, by default, under
the project root, and paid calls are capped per call by `$VENICE_MCP_MAX_SPEND` (default
**$0.10**) — **except** that `--auto` auto-approves every call and so bypasses that cap;
`--auto --assets` can incur up to `--max-tool-calls` paid generations, so use a cheap
model and a sane `--max-tool-calls` when running unattended.

Per-flag config defaults live under `defaults.code.*` (e.g. `model`, `root`, `auto`,
`assets`, `scout`, `spawn`, `spawn_max_spend`, `subagent_max_tokens`, `planner`,
`max_tool_calls`, `review`, `review_model`, `review_rounds`, `cache_guard`,
`compact_loss_policy`).

#### Scout subagent (`--scout`)

`--scout` adds one tool, `venice_scout`, so the coding agent can **delegate an
investigation to a disposable subagent instead of exploring in its own context**. The
scout starts from a **fresh context**, gets **only read-only tools**
(`read_file`/`list_dir`/`grep`/read-only `git`, plus `project_search` when a `.venice`
index exists), investigates, and returns a single **structured report** — findings,
confidence, dead-ends, what it did *not* check, and which claims it verified live vs.
inferred. Those sections are also parsed into a `fields` map on the returned report, so
the planner can consume the handoff programmatically rather than re-reading prose. The
caller sees only that report, not the dozens of tool calls behind it, so a
big "where/how does X work?" question doesn't fill the main agent's context with
exploration noise. This is a **context firewall**, not a role-specialized worker.

A scout **cannot edit files or run commands** (read-only by construction — it can never
be handed a write/exec tool, nor spawn another scout), and each run is bounded by a
tool-call budget (`max_tool_calls` in the call, default 6, hard max 15) so it can't run
away. Use it before an edit to scope the change:

```bash
venice code --scout --auto "Add rate-limiting to the API client. First scout how the \
existing client is structured and where requests are made, then implement it."
```

#### Worker subagent (`--spawn`)

Where the scout is a *read-only* context firewall, `--spawn` adds `venice_spawn` — the
same firewall for **doers**. The coding agent can **delegate a bounded implementation task
to a disposable worker** that runs in a **fresh context**, does the work, and returns a
single **structured report** — outcome, changes (files + commands), what it verified live
vs. assumed, follow-ups, and blockers. As with the scout, those sections are also parsed
into a `fields` map on the returned report, and the caller sees only the report, not the
edit churn behind it — so the planner's context stays clean while the worker does the
churny part.

A worker gets a **role-scoped subset of *your* already-built tools** (never more than the
session was granted):

- `role="code"` (default, spend-free) → `read_file`/`list_dir`/`grep`/`write_file`/
  `edit_file`/`apply_patch`/`run`/`git` (plus `project_search` when a `.venice` index
  exists) — the `fs` + `exec` + `vcs` + `search` categories.
- `role="asset"` → the media generators and their support tools (`image`/`audio`/`video`
  plus `catalog`/`vision`/`jobs`) — needs `--assets` on the parent, otherwise the grant is
  empty and the call errors.

Containment is **structural**, not a per-call prompt: a worker's writes flow through the
same **writable roots** (fail loud outside them) and the
same `run` allow/deny policy as the parent; it gets **no** `attach_root` (can't widen its
roots) and **no** scout/spawn tools (**nesting is capped at one level** — a planner
scouts/spawns, a worker does neither). Each run is bounded by a tool-call budget
(`max_tool_calls` in the call, default 12, hard max 40). An `asset` worker's cumulative
estimated **media** spend is capped in dollars by `--spawn-max-spend` (default **$2.00**,
`<= 0` disables; config `defaults.code.spawn_max_spend`); the report then also carries
`spent_usd`/`spend_cap_usd`.

Independently, `--subagent-max-tokens N` caps the cumulative **prompt+completion tokens**
a subagent spends across its turns (default off) — this applies to **both** the scout and
the worker, since token burn is universal to both, whereas the dollar cap above is
spawn-only (only a worker holds paid tools). Once the ceiling is crossed the subagent is
asked for a final answer and wraps up (the crossing turn completes, so the count can
slightly exceed the cap — like the spend cap); every report carries `tokens`/`token_cap`,
and under `--planner` the merge rollup sums `totals.tokens` and warns when a subagent hit
its cap. It's a cumulative-usage ceiling, **not** a context-window size limit, and is
distinct from `--max-tokens` (per-turn output). Config: `defaults.code.subagent_max_tokens`.

```bash
venice code --spawn --auto "Add a /health endpoint and a test for it. Spawn a code worker \
to implement and test it, then review its report before finishing."
```

#### Planner harness (`--planner`)

`--planner` turns the pieces above into one coherent workflow. It **implies
`--scout --spawn --memory`** and adds a protocol to the system prompt: **decompose** the
task into small self-contained units (`task_add` each one first), **dispatch** them
serially (`task_update` in progress → optional `venice_scout` → `venice_spawn` with the
unit's `task_id` → `task_update` done), then **merge**.

Merge is first-class, not prose: the harness records every launched scout/spawn dispatch
— its parsed report `fields`, `task_id` link, tool calls, spend, and whether it errored
or was truncated — and exposes **`venice_merge`**, a free tool that rolls all of it up
together with the [task checklist](#memory--tasks---memory) and **structural warnings**
(a task not done or never dispatched, a dispatch that errored or hit its tool-call cap,
a `task_id` that matches no task). The planner is told to resolve those warnings and end
with a `MERGE SUMMARY:` section. With `--json`, the envelope carries the same rollup
under `planner` even if the model skipped the merge call. Workers can never hold
`venice_merge` (merging is the planner's job). Dispatch is **serial by default**; add
[`--parallel`](#parallel-dispatch---parallel) to let independent units overlap.

```bash
venice code --planner --auto --json "Split the CSV importer into reader/validator/writer \
modules with tests. Decompose into units, dispatch a worker per unit, and merge."
```

#### Parallel dispatch (`--parallel`)

By default a planner dispatches one subagent at a time, so three independent units run
back-to-back — the wall-clock is the **sum** of their nested loops. `--parallel` lets the
model emit several `venice_scout`/`venice_spawn` calls in a **single turn** and runs them
**concurrently** on a bounded pool (up to 4 at once), so independent units overlap and the
wall-clock drops to roughly the **slowest single unit**. It is **opt-in** (serial
otherwise) and only affects the two subagent tools — every other tool still runs serially,
and results are stitched back in the model's original order so the transcript is
deterministic.

The prompt overlay tells the planner to dispatch units together **only when they are truly
independent** (no unit needs another's output and no two touch the same files); dependent
units stay serial. Best paired with `--planner` (it is inert without a subagent rail).
Config: `defaults.code.parallel`.

```bash
venice code --planner --parallel --auto "Add unit tests for the parser, the formatter, \
and the CLI — three independent modules. Decompose, dispatch the independent units \
together, and merge."
```

#### Reviewer rail (`--review`)

`--review` adds one tool, `venice_review`, so the coding agent can **hand its own diff
to a reviewer that did not write it** and fix the findings before handing work back. It
is the same engine as [`venice review`](#code-review-venice-review) — see there for how
the diff is scoped, how findings are formatted, and what the reviewer is and is not
allowed to do.

```bash
venice code --review --auto "Fix the retry logic in client.py, then review your own \
work and address anything the reviewer flags."
```

Two bounds worth knowing, both structural rather than prompt-enforced:

- **3 reviews per session.** A fix→review→fix spiral is the obvious failure mode of
  handing an agent a reviewer, and prose would not stop it. The 4th call is refused.
- **The model is operator-controlled.** `--review-model` is resolved once at startup and
  is *not* in the tool's schema, so the agent cannot escalate itself onto a costlier
  reviewer — the same discipline `venice_web_search` uses.

The reviewer is also unreachable from a `venice_spawn` worker (it is `category="agent"`,
which no worker role is granted), so nesting stays capped at one level.

**It cannot certify.** `venice_review` returns findings and has no way to record that a
diff was reviewed. That is deliberate: the agent holds `apply_patch` and `run`, so any
approval it *could* write it eventually *would* write — not adversarially, just because
shortest-path-to-green is ordinary agent behaviour. See the note under
[Code review](#code-review-venice-review).

## Code review (`venice review`)

A **cold-context reviewer**: a disposable subagent with a fresh context, only read-only
tools, and — where the catalog offers one — a **different model** than the one that
writes here. It reads a diff, not the repo, and reports defects.

```bash
# Review this branch (committed work AND uncommitted edits) against the default branch.
venice review

# Just what you haven't committed yet.
venice review --base HEAD

# Machine-readable, for a script or a controlling agent.
venice review --json

# Weigh one area in particular; fail the shell only on blockers.
venice review "the retry and backoff paths" --fail-on blocker
```

**Why cold context.** This is the pilot for a quality pipeline designed after two review
cycles on a sister C++ project: cold-context review of already-**merged** work found 10
confirmed bugs in one release and a shipped-unusable feature in the next — all with
tests green across 11 build configurations. Independent eyes catch what test breadth
does not. Precise findings (a repro plus `file:line`) produced a 6/6 fix rate, which is
why the report format is mandated rather than left to the model.

**What gets reviewed.** By default, `git diff -W <merge-base(default-branch, HEAD)>` —
which compares against your **working tree**, so one default covers both "review my
branch before I open the PR" and an agent's "review what I just wrote". Untracked new
files are included too (diffed against `/dev/null`), because a file that was just
created is exactly the code most worth reviewing. The default branch is auto-detected
(`origin/HEAD`, then `origin/main`, `origin/master`, `main`, `master`); override with
`--base`. `-W` is git's own function-context flag, so each hunk arrives with its
**enclosing function** in whatever language git has a driver for.

**Surface triage.** Each changed file is classified, and by default the model is only
called when the diff contains **code**:

| bucket | examples | reviewed? |
| --- | --- | --- |
| code | `src/pool.cc`, `Makefile`, anything unrecognised | yes |
| test | `tests/`, `*_test.go`, `*.spec.ts`, `conftest.py` | no |
| docs | `*.md`, `docs/`, `LICENSE`, `CHANGELOG` | no |
| generated | `package-lock.json`, `*.min.js`, `Cargo.lock` | no |
| binary | anything git reports as binary | no |

Triage runs **before** the SDK is imported or a client is built, so a docs-only diff
costs zero API calls and needs neither the `[openai]` extra nor a key. Note the tradeoff
this buys: a test-only diff that *deletes assertions* is real risk, and `auto` skips it
— use `--effort always` to review changes to tests themselves.

**Findings format.** Each defect is four lines, and the reviewer is told that a finding
it cannot locate and reproduce is not a finding yet:

```
src/pool.cc:142 [blocker] Freed buffer is reused after release()
WHY:   release() returns the slab to the freelist but the caller keeps its pointer.
REPRO: acquire(), release(), then write through the old pointer at app.cc:88 → heap UAF.
FIX:   null the caller-held pointer in release(), or return by value.
```

A missing or unrecognised severity is read as `major`, never `minor` — a quiet downgrade
past `--fail-on` is worse than a noisy over-report. A finding naming a file the diff did
not touch is reported separately and does **not** affect the exit code.

**Rounds.** `--rounds` (default 2, max 3) re-passes over the *same* diff, telling each
pass what the previous ones already reported and asking for what they missed. It stops
early as soon as a pass finds nothing new.

**Model decorrelation.** The reviewer defaults to a function-calling model from a
**different family** than the catalog default (`qwen3-4b` and `qwen-2.5-coder` count as
the same family — a different id alone is not decorrelation). If the catalog offers no
alternative it reviews with the same model and says so loudly; that is never a hard
error, but the output carries `decorrelated: false` so a caller can tell. Under `venice
code --review`, the selected reviewer id and its flag/config/automatic provenance are
reported before the first paid call; an unknown or known non-function-calling reviewer
fails at startup.

| flag | effect |
| --- | --- |
| `--base REF` | review against this ref (default: auto-detected default branch); `--base HEAD` = uncommitted only |
| `--path PATH` | limit the diff to these paths (repeatable) |
| `--model M` | reviewer model (default: a different family than the catalog default) |
| `--rounds N` | passes over the same diff (default 2, max 3); stops early when a pass adds nothing |
| `--effort auto\|always\|never` | when to spend a model call (default `auto`: only if the diff has code) |
| `--context function\|hunk` | enclosing-function context via `-W` (default) or plain hunks |
| `--fail-on none\|minor\|major\|blocker` | severity that makes the command exit 1 (default `major`) |
| `--max-diff-chars N` | cap the diff handed to the reviewer (default 60 000); files past the cap are named, never silently dropped |
| `--max-tool-calls N` | cap the reviewer's own reads beyond the diff (default 8, max 20) |
| `--subagent-max-tokens N` | cap cumulative tokens across all rounds (default off) |
| `--json` | JSON envelope (findings, verdict, `base_sha`/`head_sha`, model) to stdout |

Exit codes: **0** clean, skipped, or empty diff · **1** findings at or above `--fail-on`
· **2** not a git repo / unknown `--base` / no key · **6** unknown `--model` · **10** no
parseable verdict. Per-flag config defaults live under `defaults.review.*`.

Human output splits streams: the findings go to **stdout** (so `venice review >
findings.md` captures them and nothing else) and the provenance line — base, SHAs, model
— goes to stderr.

#### What `venice review` deliberately does not do

**It produces findings. It does not certify anything.** There is no receipt, no
signature, no approval artifact; a review writes **nothing to disk** — no session, no
state, not one byte. The exit code reports what one run found; it is not a pass, and it
is not a gate.

That is the design, not a gap. A merge gate has to live somewhere the author cannot
reach, and a coding agent holds `apply_patch` and `run` — so any receipt it *could*
write it eventually *would* write, not adversarially but because shortest-path-to-green
is ordinary agent behaviour. Fusing "find the bugs" with "declare this diff approved"
would make the second operation worthless. They are kept apart so that real enforcement
stays possible later; the `base_sha`/`head_sha` pair in the envelope exists so a future
gate can bind a decision to a specific diff.

Consistent with that, `status` distinguishes `"skipped"` from `"clean"`: "we did not
look" must never be able to masquerade as "we looked and found nothing".

## MCP server

`venice mcp-serve` runs an [MCP](https://modelcontextprotocol.io) server either
locally over stdio or remotely over authenticated Streamable HTTP. Starting the
server needs the `[mcp]` extra (Python ≥ 3.10); `venice_chat` and delegated
`venice_vision` also need `[openai]`.

### Local stdio server

The default stdio profile exposes Venice's generators as tools that a local MCP
host (Claude Code, or any process that can spawn the CLI) can call directly.
Install `[all]` when the host should be able to call all ten tools:

```sh
pip install "venice-cli[mcp]"
pip install "venice-cli[all]"       # MCP transport + chat/delegated vision

# Register it with Claude Code:
claude mcp add venice -- venice mcp-serve

# Affirm that this stdio host passes MCP ImageContent to a vision-capable model:
claude mcp add venice-vision -- venice mcp-serve --host-image-content
```

The server exposes ten tools:

| Tool | Does | Paid? |
| --- | --- | --- |
| `venice_image` | generate image(s) → file path(s) | yes (estimated) |
| `venice_tts` | synthesize speech → audio file | yes (estimated) |
| `venice_sfx` | sound effect (async queue) → audio file | yes (quoted) |
| `venice_music` | long-form music/ambience (async queue) → audio file | yes (quoted) |
| `venice_video` | text/image-to-video (async queue, long-running) → video file | yes (quoted) |
| `venice_upscale` | upscale a local image → image file | yes (dynamic) |
| `venice_bg_remove` | remove a background → transparent PNG | yes (dynamic) |
| `venice_image_edit` | edit/inpaint an image (+ optional mask layers) → image file | yes (dynamic) |
| `venice_chat` | one-shot chat completion → reply text (needs `[openai]`) | no |
| `venice_vision` | inspect a local image natively or delegate image analysis | no |

**Spend gating.** MCP is non-interactive, so instead of a `[y/N]` prompt the
paid tools gate on cost. A tool call whose estimated cost is at or under the
auto-approve cap (`VENICE_MCP_MAX_SPEND`, default **$0.10**) runs immediately.
If the estimate is over the cap — or can't be known up front, as with the
dynamically-priced `venice_upscale` / `venice_bg_remove` / `venice_image_edit` — the tool returns
`{"status": "confirmation_required", ...}` with the estimate and cap, and the
host must re-call with `confirm=true` (or a higher `max_spend`). Nothing is
spent and no file is written on a gated call. `venice_chat` is cheap and not
gated.

**Output.** Generation/editing tools write their result to a file and return its
**path**. Files land in `VENICE_MCP_OUTPUT_DIR` (default: the current working
directory), or a per-call `output_dir`. `venice_vision` is the exception: native
mode returns the authorized local image inline as MCP ImageContent. The API key
is read the usual way (`$VENICE_API_KEY` or the credentials file) and is never
echoed.

**Vision host declaration.** The default `venice mcp-serve` startup assumes a
text-only host. `venice_vision` therefore delegates in `mode=auto`, preserving
its `model` and `max_tokens` controls. `--host-image-content` is the operator's
explicit assertion that this stdio host delivers MCP ImageContent to a
vision-capable frontend: `auto` then returns a validated local `input_path`
natively, while `mode=native` requires the assertion. Remote `image_url` inputs
remain delegated; explicit native URL requests fail closed rather than adding a
server-side downloader. `mode=delegate` always preserves the text-only path. The
declaration is fixed for the process; restart or re-register the server to change
it.

**Local media inputs.** Paths supplied to image/video tools by an MCP host are
confined to the server's startup working directory after resolving symlinks.
Secret-shaped files and `.git`/`.venice` paths are refused, and the file must have
a recognized bounded image/audio/video signature before any bytes are sent. This
model-facing rail does not narrow paths explicitly supplied by an operator to the
ordinary `venice upscale`, `bg-remove`, `image-edit`, or `video` CLI commands.

Only stdout carries the JSON-RPC protocol; the server's own diagnostics go to
stderr. Video generation and image editing are exposed over MCP too: the
`venice_video` and `venice_image_edit` tools cover the same capabilities as the
`venice video` and `venice image-edit` CLI commands.

### Authenticated remote server

`venice mcp-serve --http` defaults to a deliberately small, path-independent
profile containing `venice_chat` and delegated `venice_vision`. Supplying
`--media-dir` (or `VENICE_MCP_MEDIA_DIR`) opts one server instance into the full
remote media profile: image/TTS generation, queued SFX/music/video, upscale,
background removal, image editing, import/delete, and job status/result tools.
The local stdio profile remains the ten-tool surface documented above.

Remote media never exposes a server-local path. Each result is an opaque,
principal-bound HTTPS URI returned as both JSON metadata and an MCP
`ResourceLink`. An authenticated client can read bounded objects with MCP
`resources/read`, or stream any object with `GET`/`HEAD /media/{id}` (including a
single byte range). `POST /media` accepts a raw, recognized image/audio/video body;
set `Content-Type` and optionally `X-Venice-Filename`. `venice_media_import`
supports bounded base64 image data URLs and guarded public HTTPS downloads. Paid
tools consume only media URIs owned by the current OAuth principal. A different
subject or OAuth client receives the same not-found response as a missing object.

Storage is a private SQLite metadata database plus opaque files. Objects and job
handles expire after 24 hours by default. Default limits are 100 objects and 1 GiB
per principal, 4 GiB globally, four pending jobs per principal, and 16 pending
jobs globally. Run exactly one replica against a ReadWriteOnce persistent volume;
the included Kubernetes template demonstrates that topology. Startup reconciles
expired rows and task-owned incomplete/orphaned files.

Every paid remote media call requires `confirm=true`, even below the operator's
hard `VENICE_MCP_REMOTE_MAX_SPEND` ceiling (default **$0.10**). Callers cannot
raise that ceiling. Tools whose price Venice does not disclose before execution
(`upscale`, `bg_remove`, and `image_edit`) are disabled by default; enabling them
requires both `--allow-dynamic-spend` and an explicit remote ceiling of at least
$10. This opt-in acknowledges that an unknown price cannot be checked before the
request. Chat and delegated vision retain their existing unquoted behavior.

HTTP mode is always an OAuth 2.1 resource server. It validates asymmetric JWT
access tokens locally from an external authorization server's HTTPS JWKS; Venice
does not provide a login page, issue tokens, or embed an authorization server.
Tokens must have a `kid`, use RS256, ES256, or EdDSA, and carry the configured
issuer, audience, expiry, subject, client identity (`client_id` or `azp`), and all
required scopes. A missing or invalid token is rejected before MCP parses or
invokes a tool.

```sh
export VENICE_MCP_PUBLIC_URL=https://mcp.example.com/mcp
export VENICE_MCP_OAUTH_ISSUER=https://auth.example.com
export VENICE_MCP_OAUTH_JWKS_URL=https://auth.example.com/.well-known/jwks.json
export VENICE_MCP_OAUTH_AUDIENCE=https://mcp.example.com
export VENICE_MCP_OAUTH_SCOPES='venice:mcp'

venice mcp-serve --http --host 0.0.0.0 --port 8000
```

To enable media storage:

```sh
export VENICE_MCP_MEDIA_DIR=/srv/venice-media
export VENICE_MCP_REMOTE_MAX_SPEND=0.10
venice mcp-serve --http --host 0.0.0.0 --port 8000
```

The first four values and at least one scope are required. Their equivalent CLI
flags are `--public-url`, `--oauth-issuer`, `--oauth-jwks-url`,
`--oauth-audience`, and repeatable `--oauth-scope`; explicit flags override the
environment. The public URL must be HTTPS and end exactly in `/mcp`. These
security-critical settings do not fall back to `~/.config/venice/config`.

DNS-rebinding protection accepts only the exact Host from the public URL. A
request with an Origin header is rejected unless that exact HTTPS origin was
added with repeatable `--allowed-origin` or the whitespace-separated
`VENICE_MCP_ALLOWED_ORIGINS`. This is an inbound origin check, not a wildcard
CORS switch. `GET /healthz` is intentionally unauthenticated and returns only
`{"status":"ok"}` for container probes.

Configure the external authorization server to issue access tokens for the
chosen audience and scopes, then register its OAuth client in the remote MCP host.
Claude custom connectors accept a pre-registered OAuth client ID and secret in
Advanced settings. Authentication limits who may invoke the tools; it does not add
a pre-call billing quote to `venice_chat` or delegated vision.

The published container is `ghcr.io/gobha-me/venice-cli`. Releases publish an
immutable version tag and `sha-<full-commit>` tag plus the moving `latest` tag.
Prefer the version tag or its registry digest in deployments. A hardened
Deployment, Service, and TLS Ingress template is in
`deploy/kubernetes/remote-mcp.yaml`; replace its example domain, OAuth values,
image version, TLS issuer/secret, and separately provision the referenced
`venice-mcp-api-key` Secret. Do not commit the API key to the manifest.

The reverse direction — venice as an MCP **client**, calling *other* servers'
tools inside `venice chat` — is [`venice chat --mcp`](#external-mcp-tools---mcp).

## Config

`venice config` manages a persistent, non-secret config file at
`~/.config/venice/config.json` (created mode 0600). It holds two things: an
**MCP server registry** (attached by [`venice chat --mcp`](#external-mcp-tools---mcp))
and **default flag values** so you stop repeating `--model` / `-o` / `--yes` /
`--max-spend`.

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

Global keys under `defaults` (`output_dir`, `max_spend`, `yes`, `no_balance`)
apply to any command that has the flag; a per-command section (e.g.
`defaults.chat`) overrides them. `no_balance` covers every spend-incurring
command at once -- `--show-balance` forces the display back on for a single run.
The audio **mastering chain** (`lufs`, `true_peak`, `sample_rate`, `bit_depth`,
`loop_crossfade`) is global for the same reason: `venice master`, `venice sfx
--master` and `venice music --master` share one chain, so `defaults.lufs = -14`
retargets all three at once. Any of them can still be overridden for a single
command -- `defaults.sfx.lufs = -18` beats the global on `venice sfx` only.

`output_dir` must name a directory that already **exists** — it is treated as a
directory only when it is one, so a path that doesn't exist yet is used as the
output *file* instead. (For `contact-sheet` that means an extension-less target,
which ImageMagick/ffmpeg reject.)

Those nine keys are the **complete** list of globals: everything else must be
written under its command's section. A per-command key set at the top level
(`defaults.max_wait`, `defaults.model`) is ignored rather than applied
everywhere, because "everywhere" is rarely what it means — `defaults.max_wait =
60` would have capped `venice video`, whose renders take minutes, so each one
queued, charged, and then gave up waiting.
**Precedence for any flag is: explicit CLI flag > environment variable > config
file > built-in default** — so a config default never shadows
something you pass on the command line or set in the environment.

Per-command sections cover the *persistent preferences* of most commands — the
knob is "if it expresses a preference (model, format, voice, sizing, style,
safety), it should be settable in config." Currently config-backable:

- `defaults.image.*` — `model`, `format`, `variants`, `width`, `height`,
  `aspect_ratio`, `resolution`, `style_prefix`, `preset`, `preset_file`,
  `negative_prompt`, `cfg_scale`, `steps`, `style_preset`, `style_references`,
  `embed_exif_metadata`, `lora_strength`, `quality`, `enable_web_search`,
  `disable_prompt_optimization_thinking`, `enhance_prompt`, `hide_watermark`,
  `safe_mode` (tri-state `--safe-mode`/`--no-safe-mode`; set `false` to skip
  Venice's safety blur). Note `variants` is a **cost multiplier** — it is
  reflected in the confirmation prompt before anything is charged
- `defaults.image_edit.*` — `model`, `aspect_ratio`, `resolution`, `output_format`,
  `quality`, `disable_prompt_optimization_thinking`, `enhance_prompt`, `safe_mode`
  (tri-state `--safe-mode`/`--no-safe-mode`)
- `defaults.tts.*` — `model`, `format`, `voice`, `speed`, `play`
- `defaults.sfx.*` — `model`, `duration`, `play`, `master`, `loop`,
  `no_cleanup`, `poll_interval`, `max_wait` (`loop` only takes effect when
  `master` is also on -- it is a mastering-chain knob). `model` applies to `venice sfx` only: on `venice sfx-status` the model
  identifies the queued job rather than expressing a preference, so config never
  reaches it. `venice sfx --background` prints the matching `--model` in its
  follow-up hint
- `defaults.music.*` — `model`, `duration`, `speed`, `play`, `instrumental`,
  `master`, `loop`, `no_cleanup`, `poll_interval`, `max_wait` (same `loop`
  caveat as `sfx`, and the same generate-only caveat on `model` as `sfx`)
- `defaults.master.*` — `loop`, plus a per-command override of any mastering-chain
  global above. This is the standalone `venice master` command's own section; the
  `venice sfx`/`venice music` **`--master` toggle** is the key
  `defaults.sfx.master` / `defaults.music.master`, not this table
- `defaults.video.*` — `model`, `duration`, `resolution`, `aspect_ratio`,
  `negative_prompt`, `no_audio`, `no_cleanup`, `poll_interval`, `max_wait`
- `defaults.upscale.*` — `scale`, `creativity`
- `defaults.contact_sheet.*` — `cols`, `cell`, `label`, `background`, `padding`,
  `engine`. Note the **underscore**: the command is `contact-sheet`, but config
  keys are addressed with dots, so the section is `contact_sheet` (as with
  `image_edit`). `background` here is a **color**, unlike the `--background`
  toggle on `sfx`/`music`/`video`
- `defaults.balance.*` — `min` only, a standing "warn me under $X" floor. Be
  aware this one changes an **exit code**: `venice balance` returns 1 when the
  balance is below it, so a script that never passed `--min` can start failing.
  `--json`/`--verbose` are per-invocation output modes and stay CLI-only
- `defaults.embed.*` — `model`, `dimensions`, `encoding_format`,
  `embed_base_url`, `embed_model`, `embed_ca_bundle`. `model` is the **Venice**
  catalog model and is effectively required (Venice advertises no default
  embedding model), while `embed_model` is the model name sent to an alternate
  `--embed-base-url` backend — different keys, and only one applies per run
- `defaults.review.*` — `model`, `base`, `rounds`, `effort`, `context`, `fail_on`,
  `max_diff_chars`, `max_tool_calls`, `subagent_max_tokens`, `exec_timeout`.
  Setting `model` is the usual way to make reviewer/author decorrelation permanent
- `defaults.vision.*` — `mode` (`auto`, `native`, or `delegate`), delegated
  fallback `model`, and delegated `max_tokens`. The per-image `prompt` remains a
  tool-call input rather than a standing preference
- `defaults.chat.*`, `defaults.code.*`, `defaults.index.*`,
  `defaults.search.*` — see each command's section above

Per-invocation flags (`--dry-run`, `--json`, `--resume`, `--seed`, inputs and
positionals) stay CLI-only by design.

Config enrollment is an explicit, fail-closed allow-list. The CLI does not
derive `defaults.*` keys from argparse or auto-enroll every flag except a
denylist: a new flag remains CLI-only until its persistence, coercion, and
precedence are deliberately reviewed. The test suite inventories every command
and every option on config-aware commands, so an unclassified addition fails CI
instead of silently gaining config authority.

These per-command defaults also apply when a generator runs as an **agent tool**
inside `venice chat --tools` and `venice code` — e.g. `defaults.image.safe_mode`
is honored when the model calls `venice_image`, not just on the `venice image`
CLI. `defaults.image_edit.*`, `defaults.music.instrumental`,
`defaults.video.no_audio` and `defaults.upscale.enhance` reach the tools the same
way, as do the generation knobs `defaults.image.{model,format,variants}`,
`defaults.tts.{model,format}`, `defaults.sfx.{model,duration}`,
`defaults.music.model`, `defaults.video.duration`, `defaults.upscale.scale` and
`defaults.{sfx,music,video}.max_wait`. An explicit argument the model puts in
the tool call still wins over config. `venice mcp-serve` threads the same
defaults into its wrappers.

`defaults.vision.*` applies to `venice_vision` in chat, code, and `mcp-serve`.
On the MCP server, explicit tool arguments still beat config, and native output
still requires the process-level `--host-image-content` declaration.

Not everything crosses over, and the gaps are deliberate: `poll_interval` is
CLI-only because the tool implementations fix their own polling cadence, and
`contact-sheet`, `balance` and `master` have no agent-tool surface at all, so
their sections apply only on the command line.

**Any** config value whose flag has a fixed set of choices is validated against
that set — `defaults.image.{format,quality}`,
`defaults.sfx.model`,
`defaults.image_edit.{aspect_ratio,output_format,quality}`,
`defaults.chat.web_search`,
`defaults.bit_depth` and `defaults.contact_sheet.engine`.
An unrecognized value is reported on stderr, naming the legal values exactly as
`--flag` would, and skipped — so the command falls back to its built-in default
rather than sending something the API will reject. This applies on the CLI and on
the agent-tool path alike; config must never be able to set a value the command
line would refuse. Video duration, resolution, and aspect ratio are different:
their legal values vary by model, so
`defaults.video.{duration,resolution,aspect_ratio}` are validated against the
selected model's live catalog entry before any quote or paid request.

The **API key is never stored here** — it stays in
`~/.config/venice/credentials`. Unknown keys are preserved on write, so the
schema is forward-compatible.

## Browse the model catalog

```sh
venice models                          # count by type
venice models --type music             # list ids, one per line
venice models --type music --detail    # ids + name + pricing + cache + capabilities
venice models elevenlabs-sound-effects-v2   # full JSON for one model
venice models --type all --json        # everything, raw
```

The catalog is live and may add or retire models without a CLI release. Use
`--type`, `--type all`, or a slug lookup instead of relying on a frozen count.

### Exit codes

| exit | meaning |
|---|---|
| 0 | success |
| 1 | user declined / aborted / insufficient balance (402) |
| 2 | bad input, no API key, missing prompt, stub command |
| 3 | content policy block (422) |
| 4 | rate limit (429) |
| 5 | Venice 5xx |
| 6 | not found — job not found/expired (404), or a model missing from the catalog (unknown `--model`, or no default advertised) |
| 7 | poll timeout |
| 8 | network / connection error |
| 9 | disk write error |
| 10 | verdict unparseable / ambiguous — acceptance (`venice code`) or review (`venice review`) |
| 130 | Ctrl-C |
| 141 | stdout pipe closed by the downstream reader (SIGPIPE-equivalent producer exit) |

`venice review` also uses **1** for "findings at or above `--fail-on` were reported".
That reports what a run found; it is not a certification — see
[what `venice review` deliberately does not do](#what-venice-review-deliberately-does-not-do).

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
| `VENICE_EMBED_BASE_URL` | `embed` alternate OpenAI-compatible endpoint (local backend) |
| `VENICE_EMBED_API_KEY` | key for `VENICE_EMBED_BASE_URL` (if the backend needs one) |
| `VENICE_EMBED_CA_BUNDLE` | CA bundle to trust for a self-signed embedding backend (`embed`, `index`, `search`, and the `project_search` agent tool) |
| `VENICE_MCP_MAX_SPEND` | `mcp-serve` auto-approve cap in USD (default `0.10`) |
| `VENICE_MCP_OUTPUT_DIR` | where `mcp-serve` tools write files (default: cwd) |
| `VENICE_MCP_MEDIA_DIR` | enable authenticated remote media in this private persistent directory |
| `VENICE_MCP_MEDIA_TTL_SECONDS` | remote object/job lifetime (default `86400`) |
| `VENICE_MCP_MEDIA_MAX_OBJECTS` | remote objects plus reservations per principal (default `100`) |
| `VENICE_MCP_MEDIA_PRINCIPAL_MAX_BYTES` | stored/reserved bytes per principal (default `1073741824`) |
| `VENICE_MCP_MEDIA_GLOBAL_MAX_BYTES` | stored/reserved bytes across principals (default `4294967296`) |
| `VENICE_MCP_MEDIA_MAX_PENDING_JOBS` | pending SFX/music/video jobs per principal (default `4`) |
| `VENICE_MCP_MEDIA_GLOBAL_MAX_PENDING_JOBS` | pending jobs across principals (default `16`) |
| `VENICE_MCP_MEDIA_MCP_READ_MAX_BYTES` | maximum object returned inline by MCP `resources/read` (default `33554432`) |
| `VENICE_MCP_REMOTE_MAX_SPEND` | hard known-price ceiling for one remote media call (default `0.10`) |
| `VENICE_MCP_REMOTE_ALLOW_DYNAMIC_SPEND` | explicitly enable unknown-price remote media tools (default false; ceiling must be at least `10`) |
| `VENICE_USAGE_RAW` | set to `1`/`true`/`yes`/`on` to echo each API response's raw `usage` block to **stderr** as one `usage-raw: {...}` line (diagnostic; stdout stays machine-readable) |

## Commands at a glance

| command | does what |
|---|---|
| `venice login [--embed]` | store the Venice API key (interactive, hidden, mode 0600); `--embed` stores the embed-backend key in the secret store instead |
| `venice secret set/ls/rm NAME` | manage named secrets in `~/.config/venice/secrets.json` (0600); values never printed, `ls` shows lengths only |
| `venice balance [--verbose\|--json\|--min N]` | current USD + DIEM balance |
| `venice models [--type T] [--detail] [SLUG]` | browse the catalog |
| `venice cache-probe [--model M ...] [--prefix-tokens N ...] [--repeat N] [--json] [--max-spend USD]` | controlled live prefix-cache diagnostic with raw usage details and a pre-spend confirmation gate |
| `venice sfx PROMPT [--duration N] [--max-spend USD] [...]` | generate a sound effect |
| `venice sfx-status QUEUE_ID` | fetch a backgrounded SFX job |
| `venice tts TEXT [--voice V] [--format F] [--speed N] [...]` | synthesize speech (sync) |
| `venice image PROMPT [--variants N] [--name NAME] [--max-spend USD] [...]` | generate image(s) (sync) |
| `venice image --from-file PATH [...]` | batch-generate a card set |
| `venice music PROMPT [--duration N] [--master] [--loop] [...]` | generate long-form ambience/music |
| `venice video PROMPT [--duration 5s] [--resolution R] [--aspect-ratio A] [--image F] [--reference-image F ...] [--element JSON] [...]` | generate a video (async queue, mp4); text- or image-to-video with reference inputs |
| `venice video-status QUEUE_ID [--download-url URL]` | fetch a backgrounded video job (`--download-url` is legacy-only) |
| `venice master INPUT [--loop] [--lufs N] [--bit-depth N] [...]` | master audio to WAV (48k/24-bit, LUFS/true-peak) |
| `venice contact-sheet DIR_OR_GLOB [--cols N] [--cell WxH] [--label\|--no-label] [...]` | tile images into one contact sheet (no API call) |
| `venice chat MESSAGE [--system S] [--model M] [--web-search on] [...]` | one-shot chat completion (OpenAI SDK) |
| `venice chat [-i] [--continue\|--resume ID\|FILE] [--ephemeral]` | interactive multi-turn REPL (auto-saved sessions, `/`-commands, transcripts) |
| `venice sessions ls\|show\|rm [ID]` | list/inspect/remove auto-saved chat & code sessions (`~/.config/venice/sessions/`, 0600) |
| `venice embed [TEXT] [--from-file PATH] [--model M] [--dimensions N] [--json] [--embed-base-url URL --embed-model M [--embed-ca-bundle PATH \| --embed-insecure]]` | text embeddings (OpenAI SDK; alt/local backend) |
| `venice index [PATH] [--model M] [--embed-base-url URL --embed-model M [--embed-ca-bundle PATH \| --embed-insecure]] [...]` / `venice search QUERY [-k N] [--json] [--embed-ca-bundle PATH \| --embed-insecure]` | build / query a local semantic index of a project tree (needs `[openai]`) |
| `venice code [TASK] [--auto\|--manual] [--plan-only] [-i] [--root DIR] [--continue\|--resume ID\|FILE] [--ephemeral] [--json] [...]` | coding agent: plan → accept → edit/run a project (needs `[openai]` + tool-calling model) |
| `venice review [FOCUS] [--base REF] [--rounds N] [--effort auto\|always\|never] [--fail-on LEVEL] [--json]` | cold-context review of the current diff (model-backed runs need `[openai]`; skipped/empty runs do not); **findings only — no gate, writes nothing** |
| `venice mcp-serve [--host-image-content] [--http ...] [--media-dir DIR]` | run the local MCP server or authenticated remote server, optionally with principal-bound media (needs `[mcp]`; delegated vision and `venice_chat` additionally need `[openai]`) |
| `venice config add\|list\|remove\|show` | manage the MCP server registry |
| `venice config get\|set\|unset KEY [VALUE]` | manage default flag values |
| `venice completion bash\|zsh` | print a shell tab-completion script (generated from the parser) |

## Tests

```sh
make test     # everything, including the drive suite
make drive    # the drive suite + its fake-API fixture
make openapi-check  # offline check of implemented API contracts
```

Stdlib `unittest` only. Most tests mock `urlopen` (and the OpenAI client on
model-backed paths) and patch `HOME` to a tmpdir -- no live API calls, no real
disk writes outside the tmpdir. Tests for the `[openai]` inventory above need
the OpenAI SDK importable (`pip install -e ".[openai]"`); the complete suite
uses `pip install -e ".[all,test]"`.

On top of that, `tests/test_drive_cli.py` drives the **real** CLI over a pty
with `pexpect`, pointed at a local fake API via `$VENICE_BASE_URL`. It asserts
on actual terminal output, prompts, and exit codes -- including two interleaved
multi-step dialogues (the chat REPL and the `venice code` plan gate), because
that's where interactive breakage hides. It needs the test extra
(`pip install -e ".[all,test]"`); without it the pty cases skip cleanly under
`make test`, while `make drive` tells you the dep is missing rather than
reporting a green run of nothing.

The `venice review` tests build throwaway **git repositories** with real commits,
because the reviewer shells out to git and a mocked git would only test the mock --
`-z` rename records and `-W` function context are exactly what a fake gets wrong.
They skip cleanly where no `git` binary is on `PATH`, and neutralize
`GIT_CONFIG_GLOBAL`/`GIT_CONFIG_SYSTEM` so they can never read your gitconfig.

The committed OpenAPI lock covers only operations this project intentionally
wraps. `make openapi-check` validates it without network access. Maintainers can
install `scripts/openapi-requirements.txt` and run `make openapi-live` to compare
against the official Swagger, or `make openapi-refresh` after reviewing upstream
drift. A weekly/manual workflow reports implemented-operation changes while
treating new unsupported endpoints as informational.

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

### Secrets (`venice secret`)

Named secrets *other than* the main Venice key -- currently the embed-backend key,
later MCP/cluster tokens -- live in a structured **`~/.config/venice/secrets.json`**
(mode 0600, same plaintext-file model as `credentials`), so they don't have to sit in
your shell profile or plaintext `config.json`:

```sh
venice secret set embed     # hidden prompt; stores the 'embed' secret
venice secret ls            # names + lengths only, never values
venice secret rm embed
venice login --embed        # convenience alias for `secret set embed`
```

A secret is only ever read from a hidden prompt (never on `argv`) and is **never
printed back** -- there is deliberately no command that outputs a value; `ls` shows
character counts. For a name with a canonical env var (`embed` →
`$VENICE_EMBED_API_KEY`), the env var still overrides the stored value, so CI can keep
injecting it from that system's secret store. The same plaintext caveat applies:
file permissions are the only protection.

- In CI or any shared environment, prefer `$VENICE_API_KEY` (it
  overrides the file) sourced from that system's secret store, and
  don't run `venice login` there.
- The key is never logged or printed by this tool, but it is visible to
  anything reading your process environment or scrollback. Be aware of
  what's on screen when sharing a terminal.
- If a key is exposed, revoke and rotate it at
  <https://venice.ai/settings/api>.
