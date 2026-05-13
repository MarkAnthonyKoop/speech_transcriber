"""Render a Transcript to SRT / VTT / TXT / JSON / Markdown."""

from __future__ import annotations

import json
from dataclasses import asdict

from .transcribe import Transcript


def _hhmmss_ms(t: float, comma: bool = False) -> str:
    if t < 0:
        t = 0
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = t % 60
    sep = "," if comma else "."
    return f"{h:02d}:{m:02d}:{int(s):02d}{sep}{int(round((s - int(s)) * 1000)):03d}"


def to_srt(t: Transcript) -> str:
    out = []
    for i, s in enumerate(t.segments, 1):
        out.append(str(i))
        out.append(f"{_hhmmss_ms(s.start, comma=True)} --> {_hhmmss_ms(s.end, comma=True)}")
        out.append(s.text)
        out.append("")
    return "\n".join(out)


def to_vtt(t: Transcript) -> str:
    out = ["WEBVTT", ""]
    for s in t.segments:
        out.append(f"{_hhmmss_ms(s.start)} --> {_hhmmss_ms(s.end)}")
        out.append(s.text)
        out.append("")
    return "\n".join(out)


def to_txt(t: Transcript) -> str:
    return t.text + "\n"


def to_json(t: Transcript) -> str:
    payload = {
        "source": str(t.source),
        "language": t.language,
        "language_probability": t.language_probability,
        "duration": t.duration,
        "segments": [asdict(s) for s in t.segments],
    }
    return json.dumps(payload, indent=2)


def to_markdown(t: Transcript) -> str:
    """Human-friendly transcript with collapsed [mm:ss] markers per segment."""
    out = [
        f"# {t.source.name}",
        f"_lang={t.language} ({t.language_probability:.2f})  duration={t.duration:.1f}s_",
        "",
    ]
    for s in t.segments:
        mm = int(s.start // 60)
        ss = int(s.start % 60)
        out.append(f"**[{mm:02d}:{ss:02d}]** {s.text}")
    out.append("")
    return "\n".join(out)


WRITERS = {
    "srt": to_srt,
    "vtt": to_vtt,
    "txt": to_txt,
    "json": to_json,
    "md": to_markdown,
}
