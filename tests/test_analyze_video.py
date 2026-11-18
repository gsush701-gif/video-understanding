"""
Tests for analyze_video.py.

  python3 -m unittest discover -s tests -v

Fast tests use a generated clip and a fake Whisper (no downloads).
The real-speech test runs only when RUN_REAL=1, espeak-ng is installed and the
'tiny' model can be downloaded (GitHub Actions does all three).
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "skills" / "video-understanding" / "scripts" / "analyze_video.py"
FAKE = Path(__file__).resolve().parent / "fake"
sys.path.insert(0, str(SCRIPT.parent))
import analyze_video as av  # noqa: E402


def make_clip(path, seconds=24):
    """Three visually distinct 8-second scenes plus a tone."""
    part = seconds // 3
    subprocess.run([
        "ffmpeg", "-y", "-v", "error",
        "-f", "lavfi", "-i", f"testsrc=d={part}:s=640x360:r=25",
        "-f", "lavfi", "-i", f"color=c=red:d={part}:s=640x360:r=25",
        "-f", "lavfi", "-i", f"smptebars=d={part}:s=640x360:r=25",
        "-f", "lavfi", "-i", f"sine=f=440:d={seconds}",
        "-filter_complex", "[0][1][2]concat=n=3:v=1:a=0[v]",
        "-map", "[v]", "-map", "3", "-shortest", str(path)], check=True)


def run(args, fake=True):
    env = dict(os.environ)
    if fake:
        env["PYTHONPATH"] = str(FAKE)
    p = subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True, env=env)
    if p.returncode != 0:
        raise AssertionError(f"exit {p.returncode}\nSTDERR:\n{p.stderr}")
    return json.loads(p.stdout), p.stderr


@unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg not installed")
class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        cls.clip = cls.tmp / "clip.mp4"
        make_clip(cls.clip)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_full_run_outputs(self):
        out = self.tmp / "full"
        summary, _ = run([str(self.clip), "--out", str(out)])
        for f in ["index.md", "transcript.txt", "transcript.srt", "transcript.json"]:
            self.assertTrue((out / f).exists(), f)
        self.assertEqual(summary["segments"], 6)
        self.assertGreaterEqual(summary["frames"], 3)
        self.assertTrue(summary["sheets"])
        self.assertIn("[00:00:04] Line 1 spoken here.", (out / "transcript.txt").read_text())
        self.assertIn("00:00:00,000 --> 00:00:03,500", (out / "transcript.srt").read_text())

    def test_scene_cuts_are_keyframes(self):
        out = self.tmp / "frames"
        run([str(self.clip), "--mode", "frames", "--out", str(out)])
        names = [p.name for p in (out / "frames").glob("*.jpg")]
        # cuts at 8 s and 16 s must each produce a keyframe
        self.assertTrue(any("00-00-08" in n for n in names), names)
        self.assertTrue(any("00-00-16" in n for n in names), names)
        self.assertFalse((out / "transcript.txt").exists())

    def test_static_video_is_not_oversampled(self):
        out = self.tmp / "sparse"
        summary, _ = run([str(self.clip), "--mode", "frames", "--out", str(out)])
        self.assertLessEqual(summary["frames"], 6)

    def test_start_end_offsets_timestamps(self):
        out = self.tmp / "range"
        run([str(self.clip), "--start", "2", "--end", "20", "--out", str(out)])
        first = (out / "transcript.txt").read_text().splitlines()[0]
        self.assertTrue(first.startswith("[00:00:02]"), first)

    def test_budget_mode_resumes_until_done(self):
        out = self.tmp / "budget"
        args = [str(self.clip), "--budget", "0.001", "--out", str(out)]
        statuses, runs = [], 0
        while runs < 20:
            summary, _ = run(args)
            statuses.append(summary["status"])
            runs += 1
            if summary["status"] == "done":
                break
        self.assertEqual(statuses[-1], "done")
        self.assertIn("partial", statuses)              # it really did stop and resume
        lines = (out / "transcript.txt").read_text().splitlines()
        starts = [l[:10] for l in lines]
        self.assertEqual(len(starts), len(set(starts)))  # no duplicated segments
        self.assertGreaterEqual(len(lines), 5)
        self.assertTrue(starts == sorted(starts))
        # nothing temporary is ever written into the output folder
        self.assertEqual(sorted(p.name for p in out.iterdir() if p.name.startswith(".")), [])
        self.assertTrue((out / "index.md").exists() and summary["sheets"])

    def test_budget_mode_never_deletes_in_output_folder(self):
        # Claude desktop app connected folders forbid deletes; budget mode must still finish
        out = self.tmp / "nodelete"
        env_flag = {"VIDEO_PROGRESS_DIR": str(self.tmp / "progress")}
        for _ in range(20):
            p = subprocess.run([sys.executable, str(SCRIPT), str(self.clip), "--budget", "0.001", "--out", str(out)],
                               capture_output=True, text=True,
                               env={**os.environ, "PYTHONPATH": str(FAKE), **env_flag})
            self.assertEqual(p.returncode, 0, p.stderr)
            if json.loads(p.stdout)["status"] == "done":
                break
        self.assertTrue((out / "index.md").exists())
        self.assertFalse(any(p.name.startswith(".") for p in out.iterdir()))

    def test_prompt_echo_is_dropped(self):
        out = self.tmp / "prompt"
        _, err = run([str(self.clip), "--mode", "transcript", "--prompt", "Alpha, Beta", "--out", str(out)])
        self.assertNotIn("Glossary", (out / "transcript.txt").read_text())
        self.assertIn("dropped 1 segment", err)

    def test_language_flags_reach_whisper(self):
        # run in-process so the fake module can record kwargs
        sys.path.insert(0, str(FAKE))
        import faster_whisper  # the fake
        wav = self.tmp / "a.wav"
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "sine=d=2",
                        "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(wav)], check=True)
        ns = av.argparse.Namespace(backend="faster", model="tiny", language=None, fast=False, words=False,
                                   translate=True, mixed=True, prompt=None, start=None)
        av.transcribe(wav, ns)
        kw = faster_whisper.LAST_KW
        self.assertEqual(kw["task"], "translate")
        self.assertTrue(kw["multilingual"])
        self.assertNotIn("_batched", kw)  # complete (sequential) mode by default
        # regression: Whisper must get decoded samples, never a path (PyAV API breaks)
        audio_in = faster_whisper.LAST_INPUT[0]
        self.assertTrue(hasattr(audio_in, "dtype"), type(audio_in))
        self.assertEqual(str(audio_in.dtype), "float32")
        self.assertAlmostEqual(len(audio_in) / 16000, 2.0, delta=0.1)


class HelperTests(unittest.TestCase):
    def test_timestamps(self):
        self.assertEqual(av.ts(3725.5), "01:02:05")
        self.assertEqual(av.ts(1.25, srt=True), "00:00:01,250")

    def test_said_near(self):
        segs = [{"start": 0, "end": 2, "text": "hello"}, {"start": 30, "end": 32, "text": "far"}]
        self.assertEqual(av.said_near(segs, 1), "hello")


@unittest.skipUnless(os.environ.get("RUN_REAL") == "1" and shutil.which("espeak-ng") and shutil.which("ffmpeg"),
                     "set RUN_REAL=1 with espeak-ng installed to run real-speech test")
class RealSpeechTest(unittest.TestCase):
    def test_real_transcription(self):
        tmp = Path(tempfile.mkdtemp())
        wav = tmp / "speech.wav"
        subprocess.run(["espeak-ng", "-s", "140", "-w", str(wav),
                        "Hello world. This is a test of the video understanding plugin."], check=True)
        clip = tmp / "speech.mp4"
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "color=c=blue:s=320x240:d=6",
                        "-i", str(wav), "-shortest", "-c:v", "libx264", "-c:a", "aac", str(clip)], check=True)
        summary, _ = run([str(clip), "--mode", "transcript", "--model", "tiny", "--language", "en",
                          "--out", str(tmp / "out")], fake=False)
        text = (tmp / "out" / "transcript.txt").read_text().lower()
        self.assertGreater(summary["segments"], 0)
        self.assertIn("test", text)


if __name__ == "__main__":
    unittest.main()
