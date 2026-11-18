#!/usr/bin/env python3
"""
One-time setup for video-understanding.

  python3 setup.py                 # packages + ffmpeg check + 'small' model
  python3 setup.py --model large-v3-turbo
  python3 setup.py --models-dir ~/Videos/whisper-models
  python3 setup.py --check         # report only, change nothing

Models go to <models-dir>/faster-whisper-<name>/ (default ~/whisper-models),
which analyze_video.py finds automatically. If huggingface.co is blocked
(common in sandboxes), it prints the four files to download by hand.
"""
import argparse
import importlib.util
import platform
import shutil
import subprocess
import sys
from pathlib import Path

def model_files(name):
    """large-v3 family ships vocabulary.json + preprocessor_config.json; older models vocabulary.txt."""
    if name.startswith("large-v3") or name == "turbo":
        return ["config.json", "model.bin", "tokenizer.json", "vocabulary.json", "preprocessor_config.json"]
    return ["config.json", "model.bin", "tokenizer.json", "vocabulary.txt"]
REPOS = {
    "tiny": "Systran/faster-whisper-tiny", "base": "Systran/faster-whisper-base",
    "small": "Systran/faster-whisper-small", "medium": "Systran/faster-whisper-medium",
    "large-v3": "Systran/faster-whisper-large-v3",
    "large-v3-turbo": "dropbox-dash/faster-whisper-large-v3-turbo",
}
OK, BAD = "[ok] ", "[!!] "


def has(mod):
    return importlib.util.find_spec(mod) is not None


def pip_install(pkgs):
    cmd = [sys.executable, "-m", "pip", "install", "-q", *pkgs]
    if subprocess.run(cmd).returncode == 0:
        return True
    # PEP 668 managed environments (Homebrew / Debian Python)
    return subprocess.run(cmd + ["--break-system-packages"]).returncode == 0


def check_packages(fix):
    want = [("faster_whisper", "faster-whisper"), ("PIL", "pillow"), ("yt_dlp", "yt-dlp")]
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        want.append(("mlx_whisper", "mlx-whisper"))  # fastest on Apple Silicon
    missing = [pkg for mod, pkg in want if not has(mod)]
    if not missing:
        print(OK + "Python packages installed")
        return True
    if not fix:
        print(BAD + "missing packages: " + " ".join(missing))
        return False
    print("Installing: " + " ".join(missing))
    if pip_install(missing):
        print(OK + "Python packages installed")
        return True
    print(BAD + "pip failed. Try a virtualenv:\n"
          "     python3 -m venv ~/.video-env && ~/.video-env/bin/pip install " + " ".join(missing))
    return False


def check_ffmpeg():
    if shutil.which("ffmpeg") and shutil.which("ffprobe"):
        print(OK + "ffmpeg found")
        return True
    hint = {"Darwin": "brew install ffmpeg",
            "Windows": "winget install Gyan.FFmpeg   (then restart your terminal/app)"}.get(
        platform.system(), "sudo apt install ffmpeg")
    print(BAD + "ffmpeg missing. Install it with:  " + hint)
    return False


def check_model(name, models_dir, fix):
    target = models_dir / f"faster-whisper-{name}"
    files = model_files(name)
    if all((target / f).exists() for f in files):
        print(OK + f"model '{name}' at {target}")
        return True
    if not fix:
        print(BAD + f"model '{name}' not found in {models_dir}")
        return False
    repo = REPOS.get(name, name)
    print(f"Downloading model '{name}' from huggingface.co/{repo} ...")
    try:
        from huggingface_hub import snapshot_download  # installed with faster-whisper
        snapshot_download(repo, local_dir=str(target), allow_patterns=files)
        if all((target / f).exists() for f in files):
            print(OK + f"model saved to {target}")
            return True
    except Exception as e:
        print(f"     download failed: {type(e).__name__}: {str(e)[:160]}")
    target.mkdir(parents=True, exist_ok=True)
    print(BAD + "Could not download the model (huggingface.co is probably blocked here).\n"
          f"     Download these {len(files)} files in your own browser:")
    for f in files:
        print(f"       https://huggingface.co/{repo}/resolve/main/{f}")
    print(f"     and put them in:  {target}")
    return False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="small", help="tiny|base|small|medium|large-v3|large-v3-turbo (default small)")
    ap.add_argument("--models-dir", default="~/whisper-models", help="where to keep models (default ~/whisper-models)")
    ap.add_argument("--no-model", action="store_true", help="skip the model (e.g. Apple Silicon using mlx)")
    ap.add_argument("--check", action="store_true", help="only report status")
    args = ap.parse_args()

    fix = not args.check
    results = [check_ffmpeg(), check_packages(fix)]
    if not args.no_model:
        results.append(check_model(args.model, Path(args.models_dir).expanduser(), fix))
    if all(results):
        print("\nReady. Try:  python3 analyze_video.py <video file>")
    else:
        sys.exit("\nSetup incomplete - fix the [!!] items above and run again.")


if __name__ == "__main__":
    main()
