# venice

A stdlib-only Python CLI wrapping the [Venice.ai](https://venice.ai) API.
Zero install, pod-restart-survivable (lives under `~/.local/{bin,lib}`).

Ships working `venice login`, `venice sfx` (sound-effect generation),
`venice music` (long-form ambience/music), `venice tts` (text-to-speech),
`venice image` (image generation), `venice upscale` / `venice bg-remove`
(image post-processing), `venice master` (audio mastering),
`venice balance` (budget tracking), and `venice models` (catalog
browser). `chat` and `embed` are scaffolded stubs.

## Install

```sh
git clone <this repo> ~/Projects/venice
cd ~/Projects/venice
./install.sh
```

This creates two symlinks:
- `~/.local/bin/venice` -> `<repo>/bin/venice`
- `~/.local/lib/venice` -> `<repo>/src/venice`

and `~/.config/venice/` (mode 0700) for the credentials file.
`~/.local/bin` should already be on your PATH.

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
    --preset frontline --preset-file ./frontline.json
```

`--preset-file` defaults to `~/.config/venice/image_presets.json`; point it at a
file in your project to version presets alongside the assets. Format:

```json
{
  "frontline": {
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

Only `paplay` is available in this pod by default. It plays WAV
natively, but MP3 (which is Venice's default output) relies on
PulseAudio's GStreamer plumbing -- may fail silently. If it does, the
file is still saved; the CLI just won't auto-play. To get reliable
MP3 playback in-CLI:

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
| `venice master INPUT [--loop] [--lufs N] [--bit-depth N] [...]` | master audio to WAV (48k/24-bit, LUFS/true-peak) |
| `venice chat\|embed` | stubs (exit 2) for v0.x |

## Tests

```sh
make test
```

Stdlib `unittest` only. Tests mock `urlopen` and patch `HOME` to a
tmpdir -- no live API calls, no real disk writes outside the tmpdir.

## Uninstall

```sh
./uninstall.sh
```

Removes the two symlinks only. The credentials file at
`~/.config/venice/credentials` is left alone -- delete it manually if
you want.

## Security note

The API key is plaintext on disk at `~/.config/venice/credentials`
(mode 0600). There is no OS keychain in this pod, so this is the
honest baseline. The `CLAUDE.md` at the repo root tells Claude Code
not to read or echo the file -- that's convention, not crypto. If
you share your terminal or session transcript, be aware of what's in
your scrollback.
