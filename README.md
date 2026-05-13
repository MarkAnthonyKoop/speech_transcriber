# speech_transcriber

A small wrapper around [faster-whisper](https://github.com/SYSTRAN/faster-whisper) sized for the documentary workflow under `~/claude/there_is_no_homeless/`. One file in, a structured `Transcript` (with word-level timestamps) plus SRT / VTT / TXT / JSON / Markdown out.

Built on top of CTranslate2, so it transparently uses CUDA when available (this machine has CUDA 12.2 + an NVIDIA GPU) and falls back to int8 CPU otherwise. Model weights cache on `/mnt/d/` to keep the WSL VHD lean.

---

## 1. User manual

### Install (already done on this machine)

```bash
pip3 install --user faster-whisper      # already done
```

That pulls `ctranslate2`, `huggingface-hub`, `tokenizers`, and `av`. No system packages needed.

### Transcribe one file

```bash
# Default: distil-large-v3, GPU if available, word-level timestamps on,
# markdown output written next to the source file.
python3 -m speech_transcriber transcribe path/to/audio.m4a

# Force a specific model + language + JSON output to stdout
python3 -m speech_transcriber transcribe path/to/audio.wav \
    --model small --language en --format json --stdout
```

Output formats: `md` (default), `srt`, `vtt`, `txt`, `json`. The first run for each model downloads weights into `/mnt/d/cache/huggingface/` (`tiny` ~40 MB, `distil-large-v3` ~1.5 GB).

### Transcribe a directory of recordings

```bash
python3 -m speech_transcriber batch /mnt/d/there_is_no_homeless/episodes/<id>/raw/audio/ \
    --language en --format md
# writes <stem>.md next to every audio/video file; --force overwrites existing
```

### Find a moment

```bash
python3 -m speech_transcriber find raw/audio/REC001.m4a "ez pz"
# → [02:14] ez pz, here we go...
```

Useful for locating the marker phrase in a long recording without listening through.

---

## 2. Reference

### CLI subcommands

`python3 -m speech_transcriber {transcribe|batch|find} …`

| Subcommand | Args | Flags | Notes |
| --- | --- | --- | --- |
| `transcribe PATH` | — | `--model`, `--language`, `--format`, `--output`, `--stdout`, `--no-words`, `--cpu` | One file in → one transcript out |
| `batch DIR` | — | `--model`, `--language`, `--format`, `--no-words`, `--cpu`, `--force` | Recurses; one transcript per `*.{mp3,m4a,wav,flac,aac,ogg,mp4,mov,mkv,webm,avi}` |
| `find PATH NEEDLE` | — | `--model`, `--language`, `--cpu` | Case-insensitive segment-level grep |

### Default model

`distil-large-v3` — half the size of large-v3, near-identical accuracy for English speech, ~5× faster on GPU. Override with `--model {tiny,base,small,medium,large,large-v3,distil-large-v3}` or any HF model id faster-whisper accepts.

### Python API

```python
from speech_transcriber import (
    transcribe,        # path → Transcript
    find,              # Transcript, needle → iter[(start, end, text)]
    Transcript, Segment, Word,
    to_srt, to_vtt, to_txt, to_json, to_markdown,
    WRITERS,           # dict[format → callable]
)

t = transcribe("recording.m4a", model="distil-large-v3", language="en")
print(t.text)                         # full concatenated text
for s in t.segments:                  # segment-level timestamps
    print(f"{s.start:6.2f}-{s.end:6.2f}  {s.text}")
for s in t.segments:                  # word-level (if word_timestamps=True)
    for w in s.words:
        print(f"  {w.start:5.2f}  {w.text}")
```

### Dataclasses

```python
@dataclass(frozen=True)
class Word:       start: float; end: float; text: str; probability: float
@dataclass(frozen=True)
class Segment:    start: float; end: float; text: str; words: tuple[Word, ...]
                  avg_logprob: float; no_speech_prob: float
@dataclass(frozen=True)
class Transcript: source: Path; language: str; language_probability: float
                  duration: float; segments: tuple[Segment, ...]
                  text  (property → " ".join of segment texts)
```

### File system contract

| Path | Purpose |
| --- | --- |
| `/mnt/d/cache/huggingface/hub/models--Systran--faster-whisper-*/` | Model weights (downloaded once per model) |
| `<source>.<format>` | Transcript output, written next to the source unless `--output` overrides |

The cache is never auto-pruned. `du -sh /mnt/d/cache/huggingface/` if you want to see what's there; `rm -rf /mnt/d/cache/huggingface/hub/models--Systran--faster-whisper-tiny/` to evict a specific model.

---

## 3. Architecture

### Module layout

```
speech_transcriber/
├── __init__.py    re-export surface
├── __main__.py    argparse CLI: transcribe | batch | find
├── models.py      model alias + cache location + GPU detection
├── transcribe.py  faster-whisper invocation → Transcript dataclass
└── formats.py     Transcript → SRT / VTT / TXT / JSON / Markdown
```

Dependency direction is strictly bottom-up:

```
__main__ ──► transcribe ──► models
         └─► formats   ──► transcribe (dataclasses only)
```

`formats.py` only imports the dataclasses from `transcribe.py`, not the transcription function — keep it that way so a caller can format an already-parsed transcript without paying the model-load cost.

### Why these splits

- **`models.py` is the only place that knows about HuggingFace.** Cache root, env vars, model aliases — one file. If Systran changes their model naming convention, exactly one constant updates.
- **`transcribe.py` is a single thin wrapper.** It calls `WhisperModel.transcribe` and rehomes the result into our own dataclasses so the rest of the project never imports faster-whisper directly.
- **`formats.py` is pure functions.** No side effects. Easy to test, easy to extend.

The module-level `_MODEL_CACHE` dict in `transcribe.py` is deliberate — calling `transcribe()` repeatedly in the same process reuses the loaded model. Model load is the dominant cost (~5–15s on GPU, ~30s on CPU); transcription itself is real-time-or-faster.

### Future siblings

| If we need… | Build it as a sibling |
| --- | --- |
| Speaker diarization (who said what, when) | `speech_diarizer/` — wraps pyannote or similar; consumes a `Transcript` plus the source media |
| Live / streaming transcription | `speech_stream/` — different concurrency model, different deps |
| Translation alongside transcription | `speech_translator/` — whisper has translation built in but we'd want NMT as a fallback |

`speech_transcriber` stays narrow: file → transcript. The "who's talking" and "translate to X" questions are different concerns.

### Things to know if you're modifying this

1. **The HF cache root is hard-coded to `/mnt/d/cache/huggingface/`.** WSL root has space, but the user has explicitly asked us to keep heavy artifacts off C:/the VHD. Don't relocate to `~/.cache/`.
2. **DrvFs permissions warnings are cosmetic.** When the cache lives on `/mnt/d/` (DrvFs), HuggingFace can't `chmod` files and prints a "Could not set the permissions" warning per blob. Downloads still succeed. Ignore the warnings; they're not a bug to fix.
3. **GPU detection is best-effort.** `pick_compute_type` shells out to `nvidia-smi -L`. If you build inside a container without GPU access, it'll silently fall back to CPU/int8 — which is correct behavior; don't add a "GPU required" assertion.
4. **`distil-large-v3` is English-only.** If you need other languages, switch to `large-v3` (and expect ~2× the latency).
5. **No automated test suite.** Smoke test is in `CLAUDE.md`.

---

## 4. Next steps

Concrete additions, ordered by how often the documentary / music-video work
actually needs them:

1. **`--vad off|on` CLI flag.** Today `vad_filter=True` is the only path the
   CLI uses, but `bottle/` had to call the Python API directly because
   distil-large-v3's VAD strips doom-metal vocals. Add a flag; default `on`
   (keeps current behavior), `off` for music tracks. Document the music
   gotcha next to the flag.
2. **Lyric-block alignment helper.** For music tracks, the canonical
   structure is "intro / verse / chorus / bridge / outro" with embedded
   lyrics in the source mp3 (ID3 `lyrics-eng`). A `lyrics PATH` subcommand
   that takes an ID3 lyric block + the transcript and emits aligned section
   ranges would let `video_composer` consume it directly via the planned
   `lyrics_from:` field (see `video_composer` next steps §3).
3. **Speaker diarization** belongs in a `speech_diarizer/` sibling, not here
   (already noted in `CLAUDE.md` — listed again so it stays visible).
4. **github-readiness:**
   - Move `CACHE_ROOT` in `models.py:14` to `HF_HOME` env var (which
     huggingface-hub already respects); just stop overriding it when the env
     var is set.
   - LICENSE, .gitignore.
   - Pin `faster-whisper` to a known-good version in a `requirements.txt`
     (current install is whatever pip got at first run).
   - Convert the `CLAUDE.md` smoke-test recipe into `tests/smoke.sh`.
