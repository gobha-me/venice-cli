# venice

A stdlib-only Python CLI wrapping the [Venice.ai](https://venice.ai) API.
Zero install, pod-restart-survivable (lives under `~/.local/{bin,lib}`).

v0.3 ships working `venice login`, `venice sfx` (sound-effect
generation), `venice tts` (text-to-speech), `venice balance` (budget
tracking), and `venice models` (catalog browser). `chat`, `image`, and
`embed` are scaffolded stubs.

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

```sh
venice balance              # -> $26.14 USD   (single line on stdout)
venice balance --verbose    # tier, USD, DIEM, next epoch, key expiry
venice balance --json       # raw object for scripts
venice balance --min 5      # exit 1 if balance < $5 (useful in scripts)
```

The balance endpoint is also tapped automatically by `venice sfx`,
which shows current balance + estimated remaining alongside the cost
quote. Suppress with `--no-balance`. Hard-cap the spend per call with
`--max-spend USD` (refuses to queue if the quote exceeds the cap).

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
| 1 | user declined / aborted |
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
| `venice chat\|image\|embed` | stubs (exit 2) for v0.x |

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
