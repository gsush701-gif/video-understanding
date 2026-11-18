---
name: video-understanding
description: Watches and transcribes videos so Claude can answer questions about them — local files (.mp4, .mov, .mkv, .webm, screen recordings, lectures, meetings) via fast local Whisper plus scene keyframes, and YouTube links via their transcript. Use when the user asks to transcribe, summarize, "watch this video", make subtitles/SRT, take notes from a video, translate a video, find when something is said or shown, or describe what is on screen.
---

# Video Understanding

Claude cannot watch video directly. Turn the video into text and pictures it can read:
- **YouTube link** → read the transcript YouTube already has, through a browser (seconds, no downloads).
- **Video file** → `scripts/analyze_video.py` makes a timestamped Whisper transcript and keyframes at scene changes, tiled into contact sheets with timestamps. Transcription and frames run in parallel.

`<skill-dir>` below is this skill's base directory.

## A. YouTube links

Shell sandboxes usually block YouTube, so don't download it. Use a browser tool that can reach youtube.com (built-in browser or Claude in Chrome; request site access if asked).

1. Open the watch URL.
2. Read `<skill-dir>/scripts/youtube_transcript.js` and run its contents with the browser's JavaScript tool. It pauses the video, opens "Show transcript", and returns a header line (title | channel | seconds | caption tracks) plus `[m:ss] text` lines.
3. Answer from that text, citing [mm:ss]. For what is *shown* at a moment, run `const v=document.querySelector('video'); v.currentTime=SECONDS; v.pause();`, wait 1 s, and screenshot.
4. Save a file only when asked (e.g. `<folder>/youtube-transcripts/<short-title>.txt`).

Errors:
- `NO_TRANSCRIPT` → the video has no captions. Tell the user; offer to transcribe it with Whisper if they put the video file in a folder you can reach (they may only download videos they have the rights to).
- `LAYOUT_CHANGED` → YouTube changed its page. Open the transcript panel by hand (description → "Show transcript") and read it with the page-text tool.
- Header shows `(auto)` captions → auto-generated; names and numbers may be wrong, say so when it matters.

## B. Video files

### 1. Setup (first use in a session)

Run setup in the same shell that can see the video:

```bash
python3 <skill-dir>/scripts/setup.py --check   # report only
python3 <skill-dir>/scripts/setup.py           # install packages + download 'small' model
```

If ffmpeg is missing, give the user the one command it prints. If the model download fails (huggingface.co blocked), setup prints the files to download in the user's own browser and the folder to put them in; models are found automatically in any `whisper-models/` folder under a connected folder, `~/Videos`, or `~` (or `$VIDEO_MODELS_DIR`). Apple Silicon uses mlx-whisper and needs no manual model.

If the shell that can see the video can't see `<skill-dir>` (e.g. a cloud session linked to the user's computer), copy `analyze_video.py` into the user's `whisper-models/` folder once and run it from there.

### 2. Run

```bash
python3 <skill-dir>/scripts/analyze_video.py "<path>/VIDEO.mp4" [flags] --budget 150
```

**Always use `--budget 150` when your shell commands have a time limit** (most sandboxes cap a command at ~3 min, and background jobs are killed when the command ends, so `nohup ... &` does not work). The script works for 150 s, saves progress in the output folder, and prints `"status": "partial"` with a percentage. Run the **exact same command** again until it prints `"status": "done"`; each run picks up where the last stopped, with no gaps or duplicates (tested on 28 min of speech: 2,880/2,880 words across 3 runs). Tell the user the progress between runs. Without a time limit (a normal terminal), drop `--budget`.

Rough CPU speed with `small`: ~4x real time (a 30-min video ≈ 7-8 min, i.e. 3-4 budget runs); GPU or Apple Silicon is much faster. Tell the user up front how long a long video will take.

**Always pass `--prompt` when names or terms are known** — the biggest accuracy gain for names, companies and jargon. Build a comma-separated list (under ~15 terms) from the user's message, the file name and the conversation, e.g. `--prompt "Priya Raman, Kubernetes, L2 cache"`.

| Request | Flags |
|---|---|
| Transcript or subtitles only | `--mode transcript` |
| Silent screen recording | `--mode frames` |
| One part | `--start 600 --end 900` (seconds) |
| Speed over completeness | `--fast` (batched; can skip speech, never for a transcript the user needs complete) |
| Noisy room, phone, quiet speaker | `--clean`; compare word counts with a plain run and keep the fuller one |
| Known language | `--language ne` (ISO code) |
| Language switches mid-video (e.g. Nepali + English) | `--mixed` |
| English translation | `--translate` (Whisper translates to English only) |
| Word timings | `--words` |
| Fast-cut video | `--scene 0.2 --max-frames 60` |
| Max accuracy | `--model large-v3-turbo` (after `setup.py --model large-v3-turbo`), ~2-3x slower on CPU |

Output goes to `<video-name>_analysis/` next to the video: `index.md`, `transcript.txt`, `transcript.srt`, `transcript.json`, `frames/`, `sheets/`.

### 3. Understand

1. Read `index.md` — metadata plus each keyframe paired with the speech near it.
2. View the contact sheets in `sheets/` (stage them first if they're on another machine). Each tile has its timestamp.
3. Read `transcript.txt`; grep it for keywords when long.
4. Open a single frame from `frames/` to read small on-screen text, or grab any moment: `ffmpeg -v error -ss SECONDS -i VIDEO -frames:v 1 -vf scale=1280:-2 /tmp/at.jpg`.

## Languages

- **Original language:** auto-detect; add `--language xx` if the output is in the wrong language or script.
- **To English:** `--translate`. For important content, transcribe in the original language and translate the text yourself.
- **To any other language:** transcribe (or fetch the YouTube transcript), then translate it yourself, keeping [mm:ss] stamps. For subtitles, translate `transcript.srt` line by line into `transcript.<lang>.srt`.
- **Both:** one run in the original language, then translate yourself.
- `small` is good for major languages but weaker for Nepali and other lower-resource languages; if quality is poor, suggest `large-v3-turbo`.
- Devanagari output is normal for Nepali/Hindi; transliterate to Roman script yourself if asked.

## Answering

- Cite timestamps as [mm:ss] so the user can jump there.
- Say whether a detail was **said** or **shown**; mention when they disagree.
- Whisper and auto-captions misspell names; fix them when context makes the right name clear.
- Visuals come from sampled frames — offer to sample a range more densely if brief motion matters.
- Long transcripts: point to the file rather than pasting it.

See `references/speed.md` for speed numbers and tuning.
