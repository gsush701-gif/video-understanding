# video-understanding

**Give Claude eyes and ears for video.** A Claude plugin (and standalone CLI) that turns any video into something an LLM can actually read: a fast, local, timestamped transcript plus keyframes picked at scene changes, tiled into contact sheets. YouTube links skip the download entirely and read the video's own transcript in seconds.

[![tests](https://github.com/gsush701-gif/video-understanding/actions/workflows/test.yml/badge.svg)](https://github.com/gsush701-gif/video-understanding/actions/workflows/test.yml)
![license](https://img.shields.io/badge/license-MIT-blue)

```
You:    Summarize this lecture and tell me where he explains backprop
        https://youtube.com/watch?v=...
Claude: [reads the 37-min transcript in ~5 s]
        ... Backpropagation is introduced at [14:32], where ...
```

## What it does

| Input | How | Speed |
|---|---|---|
| **YouTube link** | Opens the video in a browser and reads its transcript panel | seconds, any length |
| **Video file** (mp4, mov, mkv, webm, screen recordings…) | ffmpeg → [faster-whisper](https://github.com/SYSTRAN/faster-whisper) transcript + scene-change keyframes, in parallel | ~4x real time on a 2-core CPU; much faster on GPU / Apple Silicon |

For every video file you get:

```
lecture_analysis/
├── index.md          metadata + every keyframe paired with what was said at that moment
├── transcript.txt    [00:01:23] timestamped lines
├── transcript.srt    subtitles
├── transcript.json   segments (+ word timings with --words)
├── frames/           keyframes at scene cuts (and gap fill on long static shots)
└── sheets/           3x3 contact sheets with timestamps burned in
```

Contact sheets are the trick: a 30-minute video becomes ~4 images, so Claude can "see" the whole thing without hundreds of frames.

## Features

- **Fast, local, private** — audio never leaves your machine. Auto-picks the fastest backend: mlx-whisper (Apple Silicon), faster-whisper on CUDA, or int8 CPU. Silence and music are skipped by voice-activity detection.
- **Accurate names** — `--prompt "Priya Raman, Kubernetes"` primes the model with the names and jargon to expect; Claude fills this in from context automatically.
- **~100 languages** — auto-detect, `--language ne` to force one, `--mixed` for speech that switches languages (e.g. Nepali + English), `--translate` for English output. Claude translates into any other language itself, keeping timestamps.
- **Smart frames** — ffmpeg scene detection on a 320px proxy, extra samples only on long static stretches, capped at 36 by default.
- **Partial runs** — `--start 600 --end 900` processes only the part you care about.
- **Resumable** — `--budget 150` works in time-boxed steps and resumes exactly where it stopped, so long videos work even in sandboxes that kill commands after a few minutes.
- **Blocked-network friendly** — sandboxes that can't reach huggingface.co get the exact files to download by hand; models are found automatically in any `whisper-models/` folder.

## Install

### Claude (desktop app)
1. Download `video-understanding.plugin` from the [latest release](https://github.com/gsush701-gif/video-understanding/releases/latest) and open it in Claude, or drag it into a chat and press install.
2. Install ffmpeg: `brew install ffmpeg` (macOS) · `winget install Gyan.FFmpeg` (Windows) · `sudo apt install ffmpeg` (Linux).
3. Add the folder with your videos to the chat and ask. Claude runs setup on first use.

### Claude Code
```
/plugin marketplace add gsush701-gif/video-understanding
/plugin install video-understanding@video-understanding
```

### Standalone CLI
```bash
git clone https://github.com/gsush701-gif/video-understanding
cd video-understanding/skills/video-understanding/scripts
python3 setup.py                      # packages + ffmpeg check + 'small' model
python3 analyze_video.py talk.mp4     # -> talk_analysis/
```

Requires Python 3.9+ and ffmpeg. The first run downloads the model (~480 MB for `small`, ~1.6 GB for `large-v3-turbo`).

## Try asking Claude

- "Transcribe meeting.mp4 and list the action items with timestamps"
- "Watch this screen recording and tell me when the error first appears"
- "Make Nepali subtitles for this English lecture"
- "This video mixes Nepali and English — transcribe it and translate to English"
- "Summarize this YouTube video in 5 bullet points: <link>"
- "What's written on the slides between 12:00 and 20:00?"

## CLI options

| Flag | Use |
|---|---|
| `--mode transcript \| frames \| full` | skip what you don't need (default `full`) |
| `--prompt "names, terms"` | correct spelling of names and jargon |
| `--language xx` · `--mixed` · `--translate` | language control |
| `--start S --end S` | process a range (seconds) |
| `--model tiny…large-v3-turbo` | speed/accuracy trade-off (`small` on CPU by default) |
| `--fast` | batched decoding, ~1.6x faster, may skip some speech |
| `--clean` | denoise + level audio for noisy or quiet recordings |
| `--words` | word-level timestamps |
| `--budget 150` | resumable: stop after 150 s, rerun the same command to continue |
| `--scene 0.2 --max-frames 60` | denser frames for fast-cut video |

## Measured

On a 2-core / 3 GB sandbox VM with `small` (int8):

| Test | Result |
|---|---|
| 6-min interview | 90 s, 828 words; `--prompt` fixed every proper noun |
| `--fast` on the same | 55 s but 774 words (drops speech — why it isn't the default) |
| 28-min speech, resumable `--budget 150` runs | 3 runs, 7.3 min total, 2,880/2,880 words, 319/320 numbered sentences, no gaps at the joins |
| 25-min Nepali YouTube video | full transcript (176 lines) in seconds via captions |

See [`references/speed.md`](skills/video-understanding/references/speed.md) for tuning.

## How the YouTube part works

Shell sandboxes usually block YouTube, and caption URLs come back empty without a page-issued token, so the plugin uses the one path that works reliably: a real browser tab. [`youtube_transcript.js`](skills/video-understanding/scripts/youtube_transcript.js) opens the "Show transcript" panel and reads it (supports both the current and the previous panel markup). Videos without captions return `NO_TRANSCRIPT`, and Claude falls back to Whisper if you provide the file.

## Limitations

- Visual understanding comes from sampled keyframes, not every frame. Brief motion between frames can be missed; ask Claude to sample a range more densely.
- The `small` model is weaker on lower-resource languages (e.g. Nepali); use `large-v3-turbo` when accuracy matters.
- YouTube transcript extraction depends on YouTube's page markup and will need updating when YouTube changes it.
- Only process videos you have the right to use. The plugin reads YouTube's public transcripts and does not download YouTube videos by default.

## Development

```bash
python3 -m unittest discover -s tests -v            # fast tests, fake Whisper, generated clip
RUN_REAL=1 python3 -m unittest discover -s tests    # + real speech (needs espeak-ng, downloads 'tiny')
```

CI runs both on every push.

## License

MIT © 2026 Sushil Gautam
