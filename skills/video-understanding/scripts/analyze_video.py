#!/usr/bin/env python3
"""
analyze_video.py — turn a video into things Claude can read and see.

Outputs (in --out dir):
  transcript.txt   timestamped lines  [00:01:23] text
  transcript.srt   subtitles
  transcript.json  segments (+ word timings if --words)
  frames/          keyframes picked by scene change (JPEG, timestamp in name)
  sheets/          contact sheets: grids of keyframes with timestamps burned in
  index.md         metadata + every keyframe aligned with what is being said

Transcription and frame extraction run in parallel.
Backends (first available wins, unless --backend is set):
  mlx      mlx-whisper        (Apple Silicon, fastest on Mac)
  faster   faster-whisper     (CUDA GPU or CPU int8, fast everywhere)
  openai   openai-whisper     (slow fallback)
"""
import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


# ---------- helpers ----------

def log(msg):
    print(f"[video] {msg}", file=sys.stderr, flush=True)


def run(cmd, **kw):
    return subprocess.run(cmd, check=True, capture_output=True, text=True, **kw)


def ts(sec, srt=False):
    sec = max(0.0, float(sec))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    if srt:
        return f"{int(h):02d}:{int(m):02d}:{int(s):02d},{int((s % 1) * 1000):03d}"
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d}"


def need(binary, hint):
    if not shutil.which(binary):
        sys.exit(f"Missing '{binary}'. {hint}")


def probe(path):
    out = run(["ffprobe", "-v", "error", "-print_format", "json",
               "-show_format", "-show_streams", str(path)]).stdout
    data = json.loads(out)
    fmt = data.get("format", {})
    v = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), {})
    a = next((s for s in data.get("streams", []) if s.get("codec_type") == "audio"), None)
    fps = v.get("avg_frame_rate", "0/1")
    try:
        n, d = fps.split("/")
        fps = round(float(n) / float(d), 2) if float(d) else 0
    except Exception:
        fps = 0
    return {
        "file": Path(path).name,
        "duration": float(fmt.get("duration", 0) or 0),
        "width": v.get("width"), "height": v.get("height"), "fps": fps,
        "video_codec": v.get("codec_name"),
        "has_audio": a is not None,
        "size_mb": round(int(fmt.get("size", 0) or 0) / 1e6, 1),
    }


def download(url, workdir):
    try:
        import yt_dlp  # noqa: F401
    except ImportError:
        sys.exit("yt-dlp missing. Install with: pip install yt-dlp")
    log(f"downloading {url}")
    tmpl = str(Path(workdir) / "source.%(ext)s")
    run([sys.executable, "-m", "yt_dlp", "-f", "bv*[height<=720]+ba/b[height<=720]/b",
         "--merge-output-format", "mp4", "-o", tmpl, url])
    files = sorted(Path(workdir).glob("source.*"))
    if not files:
        sys.exit("Download failed.")
    return files[0]


# ---------- transcription ----------

def pick_backend(requested):
    if requested != "auto":
        return requested
    is_apple = platform.system() == "Darwin" and platform.machine() == "arm64"
    if is_apple:
        try:
            import mlx_whisper  # noqa: F401
            return "mlx"
        except ImportError:
            pass
    try:
        import faster_whisper  # noqa: F401
        return "faster"
    except ImportError:
        pass
    try:
        import whisper  # noqa: F401
        return "openai"
    except ImportError:
        pass
    sys.exit("No transcription backend found. Run setup.sh (installs faster-whisper).")


def has_cuda():
    try:
        import ctranslate2
        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


def model_dirs():
    """Folders where a pre-downloaded model may live (for offline / blocked networks)."""
    home = Path.home()
    dirs = []
    if os.environ.get("VIDEO_MODELS_DIR"):
        dirs.append(Path(os.environ["VIDEO_MODELS_DIR"]))
    dirs += sorted((home / "mnt").glob("*/whisper-models")) if (home / "mnt").exists() else []
    dirs += [home / "Videos" / "whisper-models", home / "whisper-models"]
    return [d for d in dirs if d.is_dir()]


def local_model(name):
    """Return a local faster-whisper model folder if one is available, else None."""
    if name and Path(name).expanduser().is_dir():
        return str(Path(name).expanduser())
    best = None
    for d in model_dirs():
        for sub in sorted(d.iterdir()):
            if (sub / "model.bin").exists():
                if name is None or name in sub.name:
                    return str(sub)
                best = best or str(sub)
    return best if name is None else None


def load_wav(path):
    """Read the 16 kHz mono PCM WAV that extract_audio() wrote into a float32 array.
    Passing samples (not a path) bypasses faster-whisper's PyAV decoding, which breaks
    whenever PyAV changes its API (e.g. av>=16 removed open(metadata_errors=...))."""
    import wave
    import numpy as np
    with wave.open(str(path), "rb") as w:
        if w.getsampwidth() != 2 or w.getnchannels() != 1:
            return str(path)  # not our format; let the backend decode it
        frames = w.readframes(w.getnframes())
    return np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0


def transcribe(audio, args, offset=None, deadline=None):
    """offset: seconds added to every timestamp (where this audio starts in the video).
    deadline: time.time() after which to stop at the next segment boundary (complete=False)."""
    backend = pick_backend(args.backend)
    t0 = time.time()
    segments, info_lang, complete = [], args.language, True
    samples = load_wav(audio) if str(audio).endswith(".wav") else str(audio)

    if backend == "faster":
        from faster_whisper import WhisperModel, BatchedInferencePipeline
        cuda = has_cuda()
        model_name = local_model(args.model) or args.model or ("large-v3-turbo" if cuda else "small")
        device, ctype = ("cuda", "float16") if cuda else ("cpu", "int8")
        log(f"transcribing with faster-whisper/{model_name} on {device}")
        try:
            model = WhisperModel(model_name, device=device, compute_type=ctype,
                                 cpu_threads=os.cpu_count() or 4)
        except Exception as e:
            sys.exit(f"Could not load model '{model_name}': {e}\n"
                     "If huggingface.co is blocked, download the model files "
                     "(config.json, model.bin, tokenizer.json, vocabulary.txt from "
                     "https://huggingface.co/Systran/faster-whisper-small) into "
                     "Videos/whisper-models/faster-whisper-small/ and run again.")
        kw = dict(language=args.language, vad_filter=True,
                  word_timestamps=args.words, beam_size=1 if args.fast else 5,
                  task="translate" if args.translate else "transcribe")
        if args.mixed and not args.language:
            kw["multilingual"] = True  # re-detect language per segment (code-switching)
        if args.prompt:
            # names/jargon to expect -> spelled right (initial_prompt only: hotwords tends to get echoed)
            kw["initial_prompt"] = f"Glossary: {args.prompt}."
        # Batched mode is ~1.6x faster but drops speech (badly so with a glossary prompt),
        # so it is only used for --fast without --prompt. Sequential carries context forward.
        segs = None
        if args.fast and not args.prompt:
            try:
                pipe = BatchedInferencePipeline(model=model)
                segs, info = pipe.transcribe(samples, batch_size=16 if cuda else 8, **kw)
            except Exception:
                segs = None
        if segs is None:
            segs, info = model.transcribe(samples, **kw)
        info_lang = info.language
        for s in segs:
            seg = {"start": s.start, "end": s.end, "text": s.text.strip()}
            if args.words and s.words:
                seg["words"] = [{"start": w.start, "end": w.end, "word": w.word} for w in s.words]
            segments.append(seg)
            if deadline and time.time() > deadline:
                complete = False  # stop cleanly at a segment boundary; caller resumes later
                break

    elif backend == "mlx":
        import mlx_whisper
        model_name = args.model or "large-v3-turbo"
        repo = model_name if "/" in model_name else f"mlx-community/whisper-{model_name}"
        log(f"transcribing with mlx-whisper/{repo}")
        r = mlx_whisper.transcribe(samples, path_or_hf_repo=repo,
                                   language=args.language, word_timestamps=args.words,
                                   task="translate" if args.translate else "transcribe")
        info_lang = r.get("language")
        for s in r["segments"]:
            seg = {"start": s["start"], "end": s["end"], "text": s["text"].strip()}
            if args.words and s.get("words"):
                seg["words"] = s["words"]
            segments.append(seg)

    elif backend == "openai":
        import whisper
        model_name = args.model or "base"
        log(f"transcribing with openai-whisper/{model_name} (slow fallback)")
        m = whisper.load_model(model_name)
        r = m.transcribe(samples, language=args.language, word_timestamps=args.words,
                         task="translate" if args.translate else "transcribe")
        info_lang = r.get("language")
        for s in r["segments"]:
            segments.append({"start": s["start"], "end": s["end"], "text": s["text"].strip()})
    else:
        sys.exit(f"Unknown backend {backend}")

    if args.prompt:
        # drop segments where the model just repeated the glossary back
        terms = [t.strip().lower() for t in args.prompt.split(",") if t.strip()]
        def echoed(text):
            low = text.lower()
            return "glossary" in low or (len(terms) >= 2 and sum(t in low for t in terms) >= max(2, len(terms) * 0.6)
                                          and len(low) < len(args.prompt) * 2.5)
        before = len(segments)
        segments = [s for s in segments if not echoed(s["text"])]
        if len(segments) < before:
            log(f"dropped {before - len(segments)} segment(s) that echoed the --prompt glossary")

    offset = (args.start or 0.0) if offset is None else offset
    if offset:
        for s in segments:
            s["start"] += offset
            s["end"] += offset
            for w in s.get("words", []):
                w["start"] += offset
                w["end"] += offset
    log(f"transcribed {len(segments)} segments in {time.time() - t0:.1f}s")
    return {"backend": backend, "language": info_lang, "segments": segments, "complete": complete}


def extract_audio(src, workdir, args, name="audio.wav"):
    wav = Path(workdir) / name
    cmd = ["ffmpeg", "-y", "-v", "error"]
    if args.start:
        cmd += ["-ss", str(args.start)]
    if args.end:
        cmd += ["-to", str(args.end)]
    cmd += ["-i", str(src), "-vn"]
    if args.clean:
        # cut rumble and hiss, reduce background noise, even out quiet/loud voices
        cmd += ["-af", "highpass=f=80,lowpass=f=8000,afftdn=nf=-25,dynaudnorm=f=150:g=15"]
    cmd += ["-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(wav)]
    run(cmd)
    return wav


def write_transcript(result, out):
    segs = result["segments"]
    (out / "transcript.json").write_text(json.dumps(result, indent=1, ensure_ascii=False))
    with open(out / "transcript.txt", "w", encoding="utf-8") as f:
        for s in segs:
            f.write(f"[{ts(s['start'])}] {s['text']}\n")
    with open(out / "transcript.srt", "w", encoding="utf-8") as f:
        for i, s in enumerate(segs, 1):
            f.write(f"{i}\n{ts(s['start'], True)} --> {ts(s['end'], True)}\n{s['text']}\n\n")


# ---------- frames ----------

def scene_times(src, args):
    """Timestamps of scene changes using ffmpeg's scene score."""
    cmd = ["ffmpeg", "-v", "info", "-nostats"]
    if args.start:
        cmd += ["-ss", str(args.start)]
    if args.end:
        cmd += ["-to", str(args.end)]
    # downscale first so detection is fast
    cmd += ["-i", str(src), "-an",
            "-vf", f"scale=320:-2,select='gt(scene,{args.scene})',showinfo",
            "-f", "null", "-"]
    p = subprocess.run(cmd, capture_output=True, text=True)
    times = [float(m) for m in re.findall(r"pts_time:([0-9.]+)", p.stderr)]
    off = args.start or 0.0
    return [t + off for t in times]


def choose_times(src, meta, args):
    start = args.start or 0.0
    end = args.end or meta["duration"]
    span = max(end - start, 0.1)
    n = args.max_frames
    scenes = scene_times(src, args)
    times = [start + 0.5] + scenes
    # fill long gaps with uniform samples so nothing big is skipped
    fill_every = max(span / (n * 0.5), 10.0)
    filled = []
    for a, b in zip(times, times[1:] + [end]):
        filled.append(a)
        k = int((b - a) // fill_every)
        for j in range(1, k + 1):
            filled.append(a + j * (b - a) / (k + 1))
    times = sorted(set(round(t, 2) for t in filled if start <= t < end))
    # thin to max_frames evenly, keeping first
    if len(times) > n:
        step = len(times) / n
        times = [times[int(i * step)] for i in range(n)]
    log(f"{len(scenes)} scene changes -> {len(times)} keyframes")
    return times


def grab_frame(src, t, path, width):
    run(["ffmpeg", "-y", "-v", "error", "-ss", f"{t:.2f}", "-i", str(src),
         "-frames:v", "1", "-vf", f"scale={width}:-2", "-q:v", "3", str(path)])


def extract_frames(src, meta, out, args):
    frames_dir = out / "frames"
    frames_dir.mkdir(exist_ok=True)
    times = choose_times(src, meta, args)
    paths = []
    with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 4)) as ex:
        jobs = []
        for t in times:
            p = frames_dir / f"frame_{ts(t).replace(':', '-')}_{int((t % 1) * 100):02d}.jpg"
            jobs.append((t, p, ex.submit(grab_frame, src, t, p, args.width)))
        for t, p, j in jobs:
            try:
                j.result()
                if p.exists():
                    paths.append((t, p))
            except Exception as e:
                log(f"frame at {t:.1f}s failed: {e}")
    return paths


def contact_sheets(frames, out, cols=3, rows=3, tile_w=512):
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        log("Pillow not installed; skipping contact sheets")
        return []
    sheets_dir = out / "sheets"
    sheets_dir.mkdir(exist_ok=True)
    per = cols * rows
    sheets = []
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 26)
    except Exception:
        try:
            font = ImageFont.load_default(size=26)
        except TypeError:
            font = ImageFont.load_default()
    for si in range(0, len(frames), per):
        chunk = frames[si:si + per]
        imgs = []
        for t, p in chunk:
            im = Image.open(p).convert("RGB")
            im = im.resize((tile_w, int(im.height * tile_w / im.width)))
            imgs.append((t, im))
        tile_h = max(im.height for _, im in imgs)
        c = min(cols, len(imgs))
        r = (len(imgs) + cols - 1) // cols
        sheet = Image.new("RGB", (c * tile_w, r * tile_h), "black")
        d = ImageDraw.Draw(sheet)
        for i, (t, im) in enumerate(imgs):
            x, y = (i % cols) * tile_w, (i // cols) * tile_h
            sheet.paste(im, (x, y))
            label = ts(t)
            d.rectangle([x, y, x + 130, y + 36], fill="black")
            d.text((x + 8, y + 4), label, fill="yellow", font=font)
        name = sheets_dir / f"sheet_{si // per + 1:02d}_{ts(chunk[0][0]).replace(':', '-')}.jpg"
        sheet.save(name, quality=85)
        sheets.append((chunk[0][0], chunk[-1][0], name))
    return sheets


# ---------- index ----------

def said_near(segments, t, window=4.0):
    hits = [s["text"] for s in segments if s["start"] - window <= t <= s["end"] + window]
    txt = " ".join(hits)
    return (txt[:220] + "…") if len(txt) > 220 else txt


def write_index(out, meta, transcript, frames, sheets, elapsed):
    L = [f"# Video analysis: {meta['file']}", ""]
    L.append(f"- Duration: {ts(meta['duration'])} ({meta['duration']:.1f}s)")
    L.append(f"- Resolution: {meta['width']}x{meta['height']} @ {meta['fps']} fps")
    L.append(f"- Audio: {'yes' if meta['has_audio'] else 'none'}")
    if transcript:
        L.append(f"- Transcript: {len(transcript['segments'])} segments, "
                 f"language={transcript['language']}, backend={transcript['backend']}")
    L.append(f"- Processed in {elapsed:.1f}s")
    L.append("")
    if sheets:
        L.append("## Contact sheets (view these first)")
        for a, b, p in sheets:
            L.append(f"- {p.relative_to(out)}  covers {ts(a)} – {ts(b)}")
        L.append("")
    if frames:
        L.append("## Keyframes" + (" with speech nearby" if transcript else ""))
        segs = transcript["segments"] if transcript else []
        for t, p in frames:
            line = f"- [{ts(t)}] {p.relative_to(out)}"
            near = said_near(segs, t) if segs else ""
            if near:
                line += f" — “{near}”"
            L.append(line)
        L.append("")
    if transcript:
        L.append("## Files")
        L.append("- transcript.txt (timestamped), transcript.srt, transcript.json")
    (out / "index.md").write_text("\n".join(L), encoding="utf-8")


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="video file path or URL (YouTube etc. needs yt-dlp)")
    ap.add_argument("--out", help="output dir (default: <video>_analysis next to the video)")
    ap.add_argument("--mode", choices=["full", "transcript", "frames"], default="full")
    ap.add_argument("--backend", choices=["auto", "faster", "mlx", "openai"], default="auto")
    ap.add_argument("--model", help="whisper model (default: large-v3-turbo on GPU/Mac, small on CPU)")
    ap.add_argument("--language", help="force language code, e.g. en, ne, hi (default: auto-detect)")
    ap.add_argument("--prompt", help="names and terms likely in the video, e.g. 'Priya Raman, Kubernetes, L2 cache'")
    ap.add_argument("--translate", action="store_true", help="output English translation instead of original-language text")
    ap.add_argument("--mixed", action="store_true", help="speech switches languages (e.g. Nepali + English): detect per segment")
    ap.add_argument("--clean", action="store_true", help="denoise and level the audio first (noisy rooms, phone recordings)")
    ap.add_argument("--fast", action="store_true", help="greedy decoding: faster, slightly less accurate")
    ap.add_argument("--words", action="store_true", help="include word-level timestamps")
    ap.add_argument("--start", type=float, help="start at this second")
    ap.add_argument("--end", type=float, help="stop at this second")
    ap.add_argument("--budget", type=float,
                    help="resumable mode: stop after this many seconds, save progress, and continue "
                         "on the next identical run (for shells that kill commands after ~3 min)")
    ap.add_argument("--max-frames", type=int, default=36, help="cap on keyframes (default 36)")
    ap.add_argument("--scene", type=float, default=0.3, help="scene-change threshold 0-1 (lower = more frames)")
    ap.add_argument("--width", type=int, default=768, help="keyframe width in px")
    args = ap.parse_args()

    need("ffmpeg", "Install ffmpeg (macOS: brew install ffmpeg, Windows: winget install ffmpeg).")
    need("ffprobe", "ffprobe ships with ffmpeg.")
    t0 = time.time()
    workdir = tempfile.mkdtemp(prefix="vid_")

    src = args.input
    if re.match(r"^https?://", src):
        src = download(src, workdir)
    src = Path(src).expanduser().resolve()
    if not src.exists():
        sys.exit(f"Not found: {src}")

    out = Path(args.out).expanduser() if args.out else src.with_name(src.stem + "_analysis")
    out.mkdir(parents=True, exist_ok=True)
    meta = probe(src)
    log(f"{meta['file']}: {ts(meta['duration'])}, {meta['width']}x{meta['height']}, audio={meta['has_audio']}")

    want_tx = args.mode in ("full", "transcript") and meta["has_audio"]
    want_fr = args.mode in ("full", "frames") and meta["width"]

    if args.budget:
        run_budgeted(src, out, meta, want_tx, want_fr, args, t0, workdir)
        return

    transcript, frames, sheets = None, [], []
    with ThreadPoolExecutor(max_workers=2) as ex:
        fut_tx = ex.submit(lambda: transcribe(extract_audio(src, workdir, args), args)) if want_tx else None
        fut_fr = ex.submit(extract_frames, src, meta, out, args) if want_fr else None
        if fut_fr:
            frames = fut_fr.result()
            sheets = contact_sheets(frames, out)
        if fut_tx:
            transcript = fut_tx.result()
            write_transcript(transcript, out)

    finish(out, meta, transcript, frames, sheets, t0, workdir)


def finish(out, meta, transcript, frames, sheets, t0, workdir):
    elapsed = time.time() - t0
    write_index(out, meta, transcript, frames, sheets, elapsed)
    shutil.rmtree(workdir, ignore_errors=True)
    log(f"done in {elapsed:.1f}s -> {out}")
    print(json.dumps({"status": "done", "out_dir": str(out), "index": str(out / "index.md"),
                      "sheets": [str(p) for _, _, p in sheets],
                      "frames": len(frames),
                      "segments": len(transcript["segments"]) if transcript else 0,
                      "seconds": round(elapsed, 1)}, indent=1))


def run_budgeted(src, out, meta, want_tx, want_fr, args, t0, workdir):
    """Resumable mode for shells that kill long commands: work for --budget seconds,
    save progress in <out>/.progress.json, print status "partial", and continue from
    there when the exact same command is run again."""
    # Progress lives in the temp dir, not the output folder: some sandboxes (e.g. the
    # Claude desktop app's connected folders) forbid deleting files there, and the
    # audio cache can be large. Keyed by output folder so reruns find it.
    import hashlib
    key = hashlib.sha1(str(out.resolve()).encode()).hexdigest()[:16]
    pdir = Path(os.environ.get("VIDEO_PROGRESS_DIR") or tempfile.gettempdir()) / "video-understanding" / key
    pdir.mkdir(parents=True, exist_ok=True)
    state_p, cache = pdir / "progress.json", pdir / "audio.wav"
    st = json.loads(state_p.read_text()) if state_p.exists() else {}
    if st.get("source") not in (None, str(src)):
        st = {}
    st["source"] = str(src)
    deadline = t0 + args.budget
    v_start = args.start or 0.0
    v_end = args.end or meta["duration"]

    frames, sheets = [], []
    with ThreadPoolExecutor(max_workers=2) as ex:
        fut_fr = None
        if want_fr and not st.get("frames"):
            fut_fr = ex.submit(extract_frames, src, meta, out, args)
        if want_tx and not st.get("tx_done"):
            if not cache.exists():
                extract_audio(src, pdir, args, name="audio.wav")
            done = st.get("done_until", v_start)
            rest = Path(workdir) / "rest.wav"
            run(["ffmpeg", "-y", "-v", "error", "-ss", f"{done - v_start:.3f}", "-i", str(cache), str(rest)])
            tx = transcribe(rest, args, offset=done, deadline=deadline)
            st["segments"] = st.get("segments", []) + tx["segments"]
            st["language"], st["backend"] = tx["language"], tx["backend"]
            if tx["complete"]:
                st["tx_done"] = True
            elif tx["segments"]:
                st["done_until"] = tx["segments"][-1]["end"]
        if fut_fr:
            st["frames"] = [[t, str(p)] for t, p in fut_fr.result()]
    st["frames"] = st.get("frames", [])
    state_p.write_text(json.dumps(st))

    if want_tx and not st.get("tx_done"):
        done = st.get("done_until", v_start)
        span = max(v_end - v_start, 1)
        pct = round(100 * (done - v_start) / span)
        log(f"partial: transcribed up to {ts(done)} of {ts(v_end)} ({pct}%) - run the same command again")
        shutil.rmtree(workdir, ignore_errors=True)
        print(json.dumps({"status": "partial", "done_until": round(done, 1), "end": round(v_end, 1),
                          "percent": pct, "segments_so_far": len(st["segments"]),
                          "next": "run the exact same command again to continue"}, indent=1))
        return

    frames = [(t, Path(p)) for t, p in st["frames"]]
    if frames:
        sheets = contact_sheets(frames, out)
    transcript = None
    if want_tx:
        transcript = {"backend": st["backend"], "language": st["language"],
                      "segments": st["segments"], "complete": True}
        write_transcript(transcript, out)
    shutil.rmtree(pdir, ignore_errors=True)
    finish(out, meta, transcript, frames, sheets, t0, workdir)


if __name__ == "__main__":
    main()
