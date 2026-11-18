# Speed and accuracy tuning

Rough transcription speed (audio minutes per wall-clock minute). Real numbers vary with hardware.

| Hardware | Backend / model | Approx. speed |
|---|---|---|
| NVIDIA GPU | faster-whisper large-v3-turbo, batched fp16 | 50–100x real time |
| Apple Silicon (M1–M4) | mlx-whisper large-v3-turbo | 10–30x |
| Modern CPU, 8+ cores | faster-whisper small, int8 | 5–15x |
| 2-core sandbox VM (measured) | faster-whisper small, int8 | ~4x (6-min video in 90 s) |
| Same, `--fast` (batched) | faster-whisper small, int8 | ~6x |
| Any | openai-whisper (fallback) | 1–3x |

Frame extraction is a single fast ffmpeg pass on a 320px downscale plus one seek per keyframe; it finishes well before transcription on almost any video.

## Levers, fastest first

1. `--start/--end` — only process the part that matters.
2. `--model base` or `tiny` — big speedup, more errors on names and jargon.
3. `--fast` — batched inference + greedy decoding. Measured on a 6-minute interview: 55 s vs 90 s, but it captured 774 words vs 828 — it drops some speech, so don't use it when a complete transcript matters (and it is disabled when `--prompt` is set, since batched + prompt dropped ~30% of the speech).
4. `--language xx` — skips language detection and prevents wrong-language output.
5. `--mode transcript` — skip frames when visuals don't matter.

VAD (voice activity detection) is always on, so silence and music are skipped rather than transcribed — this is a large speedup on lectures and meetings with pauses.

## Accuracy levers

- `--prompt "names, terms"` — fixes proper nouns (measured: company and person names went from wrong to right).

- `--model large-v3-turbo` (best speed/accuracy balance) or `--model large-v3` (most accurate, slowest).
- Drop `--fast`.
- Set `--language` for non-English or code-switched speech (e.g. Nepali/English).
