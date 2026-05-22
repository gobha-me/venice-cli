# venice

A stdlib-only Python CLI wrapping the [Venice.ai](https://venice.ai) API.
Zero install, pod-restart-survivable (lives under `~/.local/{bin,lib}`).

v0.1 ships working `venice login` + `venice sfx` (sound-effect
generation). `chat`, `tts`, `image`, and `embed` are scaffolded stubs.

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

## Sound effects

```sh
# Quote only -- no charge, no audio.
venice sfx "thunderstorm rolling in" --duration 8 --dry-run

# Generate, confirm cost, save to ./venice-sfx-<id>.mp3.
venice sfx "soft chime" --duration 2

# Auto-confirm, custom output path, no playback.
venice sfx "rain on tin roof" --duration 4 --yes -o /tmp/rain.mp3 --no-play

# Background: prints queue_id to stdout, fetch later.
ID=$(venice sfx "ocean waves" --duration 10 --yes --background)
venice sfx-status "$ID" -o /tmp/ocean.mp3
```

### Models

| slug | max duration |
|---|---|
| `elevenlabs-sound-effects-v2` (default) | 22 s |
| `mmaudio-v2-text-to-audio` | 30 s |

Durations longer than the model max are clamped (warning on stderr).

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
