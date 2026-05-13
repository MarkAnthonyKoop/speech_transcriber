"""Core: media path → list of transcription segments.

We use faster-whisper directly (not the openai-whisper reference impl) because
it's an order of magnitude faster on CPU and works with CUDA. Input is any
file ffmpeg can read — faster-whisper invokes ffmpeg under the hood.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from .models import (
    DEFAULT_MODEL,
    configure_cache,
    pick_compute_type,
    resolve_model,
)


@dataclass(frozen=True)
class Word:
    start: float
    end: float
    text: str
    probability: float = 0.0


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    text: str
    words: tuple[Word, ...] = field(default_factory=tuple)
    avg_logprob: float = 0.0
    no_speech_prob: float = 0.0


@dataclass(frozen=True)
class Transcript:
    source: Path
    language: str
    language_probability: float
    duration: float
    segments: tuple[Segment, ...]

    @property
    def text(self) -> str:
        return " ".join(s.text.strip() for s in self.segments).strip()


_MODEL_CACHE: dict[tuple[str, str, str], object] = {}


def _get_model(name: str, device: str, compute_type: str):
    key = (name, device, compute_type)
    if key not in _MODEL_CACHE:
        from faster_whisper import WhisperModel
        _MODEL_CACHE[key] = WhisperModel(name, device=device, compute_type=compute_type)
    return _MODEL_CACHE[key]


def transcribe(
    path: Path | str,
    *,
    model: str = DEFAULT_MODEL,
    language: str | None = None,
    word_timestamps: bool = True,
    prefer_gpu: bool = True,
    vad_filter: bool = True,
    beam_size: int = 5,
) -> Transcript:
    configure_cache()
    model_name = resolve_model(model)
    device, compute_type = pick_compute_type(prefer_gpu=prefer_gpu)
    whisper = _get_model(model_name, device, compute_type)

    segments_iter, info = whisper.transcribe(
        str(path),
        language=language,
        beam_size=beam_size,
        word_timestamps=word_timestamps,
        vad_filter=vad_filter,
    )

    segs: list[Segment] = []
    for s in segments_iter:
        words = tuple(
            Word(start=w.start, end=w.end, text=w.word.strip(), probability=getattr(w, "probability", 0.0))
            for w in (s.words or ())
        ) if word_timestamps else ()
        segs.append(Segment(
            start=s.start, end=s.end, text=s.text.strip(),
            words=words,
            avg_logprob=getattr(s, "avg_logprob", 0.0),
            no_speech_prob=getattr(s, "no_speech_prob", 0.0),
        ))

    return Transcript(
        source=Path(path),
        language=info.language,
        language_probability=info.language_probability,
        duration=info.duration,
        segments=tuple(segs),
    )


def find(transcript: Transcript, needle: str) -> Iterator[tuple[float, float, str]]:
    """Yield (start, end, segment_text) for every segment containing `needle` (case-insensitive)."""
    n = needle.lower()
    for s in transcript.segments:
        if n in s.text.lower():
            yield s.start, s.end, s.text
