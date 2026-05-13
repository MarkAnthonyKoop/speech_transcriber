# CLAUDE.md — speech_transcriber

Instructions for any AI (or human) modifying this package. Read `README.md` first. Universal rules live in `~/CLAUDE.md`.

## We do one thing: file → transcript

This package does not:

- Diarize speakers — that's a future `speech_diarizer/`.
- Stream live audio — that's a future `speech_stream/`.
- Translate — whisper *can*, but the use case for our documentary work is monolingual; if translation is needed, that's a future sibling.
- Cut, denoise, or modify audio — that's `audio_fx/`.

If you find yourself adding non-transcription logic, it belongs in a sibling.

## The cache lives on /mnt/d

Model weights are 1.5 GB+. The HF cache is hard-coded to `/mnt/d/cache/huggingface/` in `models.py`. Don't relocate to `~/.cache/`.

DrvFs (Windows mount) will print "Could not set the permissions" warnings during model downloads. **These are cosmetic.** Downloads complete fine. Don't try to "fix" the warnings — there's no fix from this side; it's a WSL/DrvFs limitation. Suppress them if they bother you, but don't add a Linux-filesystem fallback.

## Default model is distil-large-v3 — English-only

If a caller passes a non-English file with the default model, they get garbage. The `--language` flag is autodetect by default; if the autodetect comes back not-English on a model that only does English, surface a warning. Better: caller should override `--model large-v3` for non-English.

## Module dependency direction

```
__main__ ──► transcribe ──► models
         └─► formats   ──► transcribe (dataclasses only)
```

`formats.py` must not import `transcribe.transcribe` (the function). It only imports the *dataclasses* from `transcribe`. A caller with a pre-parsed transcript should be able to format it without paying for a model load.

## Files stay small

All files ≤ 150 lines (soft) / 200 (hard). No `utils.py`. `__init__.py` only re-exports.

## Models are cached in-process

`_MODEL_CACHE` in `transcribe.py` keeps `WhisperModel` instances alive for the life of the process. Don't move this to a global module-level singleton or to a config object — keep the dict private to the function.

If you batch-transcribe many files in one process, the first call pays the load cost and the rest are nearly instant. Don't reload the model per file.

## When something is broken, fix the root cause

- `RuntimeError: CUDA out of memory` → drop to a smaller model or pass `--cpu`. Don't catch + retry.
- `Model not found` → wrong alias; update `_ALIAS` in `models.py`.
- DrvFs permission warnings → ignore (see above).
- `Failed to detect devices under "/sys/class/drm/card0"` from onnxruntime → harmless; that's onnxruntime probing for Intel iGPU. Don't suppress globally.

## Documentation contract

Behavior changes update `README.md` in the same change. New CLI flags go in §2 tables. New formats go in `WRITERS` *and* the format-list line in §1.

## Smoke test before declaring done

```bash
python3 -c "import speech_transcriber; print('import ok')"
python3 -m speech_transcriber --help >/dev/null && echo "cli ok"

# Tiny model + silent input — exercises the full pipeline cheaply
mkdir -p /tmp/st_smoke
ffmpeg -y -hide_banner -loglevel error -f lavfi -i "sine=frequency=440:duration=3" \
       -c:a pcm_s16le /tmp/st_smoke/sine.wav
for fmt in md srt vtt txt json; do
    python3 -m speech_transcriber transcribe /tmp/st_smoke/sine.wav \
        --model tiny --format $fmt --stdout >/dev/null && echo "$fmt ok"
done
```

If you actually changed `transcribe.py`, run a *real* speech sample (any audiobook clip, podcast, etc.) and confirm word timestamps look reasonable.
