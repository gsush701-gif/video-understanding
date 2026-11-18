"""Stand-in for faster_whisper so the pipeline can be tested without a model download.
Records the options it was called with in LAST_KW for assertions."""
from types import SimpleNamespace as N

LAST_KW = {}


class WhisperModel:
    def __init__(self, *a, **k):
        pass

    def transcribe(self, path, **kw):
        LAST_KW.clear()
        LAST_KW.update(kw)
        import wave
        try:
            with wave.open(str(path)) as w:
                dur = w.getnframes() / w.getframerate()
        except Exception:
            dur = 24.0
        # one 3.5 s "spoken" line every 4 s of audio, like a real model would emit
        segs = [N(start=i * 4.0, end=i * 4.0 + 3.5, text=f" Line {i} spoken here.", words=None)
                for i in range(int(dur // 4))]
        if kw.get("initial_prompt"):  # simulate the model echoing the glossary once
            segs.append(N(start=24.0, end=25.0, text=" Glossary: Alpha, Beta.", words=None))
        return iter(segs), N(language="en")


class BatchedInferencePipeline:
    def __init__(self, model):
        self.model = model

    def transcribe(self, path, batch_size=8, **kw):
        kw["_batched"] = True
        return self.model.transcribe(path, **kw)
