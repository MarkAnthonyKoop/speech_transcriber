"""speech_transcriber — faster-whisper wrapper sized for documentary workflows.

One audio/video file in, a `Transcript` with word-level timestamps out. Plus
formatters for SRT/VTT/TXT/JSON/Markdown.
"""

from .transcribe import (
    Transcript,
    Segment,
    Word,
    transcribe,
    find,
)
from .formats import to_srt, to_vtt, to_txt, to_json, to_markdown, WRITERS
from .models import (
    DEFAULT_MODEL,
    CACHE_ROOT,
    configure_cache,
    pick_compute_type,
    resolve_model,
)

__all__ = [
    "Transcript",
    "Segment",
    "Word",
    "transcribe",
    "find",
    "to_srt",
    "to_vtt",
    "to_txt",
    "to_json",
    "to_markdown",
    "WRITERS",
    "DEFAULT_MODEL",
    "CACHE_ROOT",
    "configure_cache",
    "pick_compute_type",
    "resolve_model",
]
