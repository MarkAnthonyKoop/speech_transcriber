"""CLI: `python3 -m speech_transcriber <subcommand> …`"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .models import DEFAULT_MODEL
from .transcribe import transcribe, find
from .formats import WRITERS


_AUDIO_VIDEO_EXTS = {".mp3", ".m4a", ".wav", ".flac", ".aac", ".ogg",
                     ".mp4", ".mov", ".mkv", ".webm", ".avi"}


def _output_path(src: Path, fmt: str) -> Path:
    return src.with_suffix(f".{fmt}")


def _cmd_transcribe(args: argparse.Namespace) -> int:
    path = Path(args.path)
    t = transcribe(
        path, model=args.model, language=args.language,
        word_timestamps=not args.no_words, prefer_gpu=not args.cpu,
    )
    text = WRITERS[args.format](t)
    if args.stdout:
        sys.stdout.write(text)
    else:
        out = Path(args.output) if args.output else _output_path(path, args.format)
        out.write_text(text)
        print(out)
    return 0


def _cmd_batch(args: argparse.Namespace) -> int:
    root = Path(args.dir)
    files = sorted(p for p in root.rglob("*") if p.suffix.lower() in _AUDIO_VIDEO_EXTS)
    if not files:
        print(f"no audio/video found under {root}", file=sys.stderr)
        return 2
    for p in files:
        out = _output_path(p, args.format)
        if out.exists() and not args.force:
            print(f"skip {p.name} (exists)", file=sys.stderr)
            continue
        print(f"... {p}", file=sys.stderr)
        t = transcribe(p, model=args.model, language=args.language,
                       word_timestamps=not args.no_words, prefer_gpu=not args.cpu)
        out.write_text(WRITERS[args.format](t))
        print(out)
    return 0


def _cmd_find(args: argparse.Namespace) -> int:
    t = transcribe(Path(args.path), model=args.model, language=args.language,
                   word_timestamps=False, prefer_gpu=not args.cpu)
    hits = list(find(t, args.needle))
    if not hits:
        print(f"no hits for {args.needle!r}", file=sys.stderr)
        return 1
    for start, end, text in hits:
        mm, ss = int(start // 60), int(start % 60)
        print(f"[{mm:02d}:{ss:02d}] {text}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="speech_transcriber")
    sub = p.add_subparsers(dest="cmd", required=True)

    common_model = ("--model", {"default": DEFAULT_MODEL,
                                "help": f"whisper model (default: {DEFAULT_MODEL})"})
    common_lang = ("--language", {"default": None, "help": "ISO code; default = autodetect"})
    common_cpu = ("--cpu", {"action": "store_true", "help": "force CPU even if CUDA is present"})

    pt = sub.add_parser("transcribe", help="transcribe one file")
    pt.add_argument("path")
    pt.add_argument(common_model[0], **common_model[1])
    pt.add_argument(common_lang[0], **common_lang[1])
    pt.add_argument(common_cpu[0], **common_cpu[1])
    pt.add_argument("--format", choices=sorted(WRITERS), default="md")
    pt.add_argument("--output", help="output path (default: same dir, same stem)")
    pt.add_argument("--stdout", action="store_true", help="write to stdout instead of a file")
    pt.add_argument("--no-words", action="store_true", help="skip word-level timestamps")
    pt.set_defaults(func=_cmd_transcribe)

    pb = sub.add_parser("batch", help="transcribe every audio/video file under a directory")
    pb.add_argument("dir")
    pb.add_argument(common_model[0], **common_model[1])
    pb.add_argument(common_lang[0], **common_lang[1])
    pb.add_argument(common_cpu[0], **common_cpu[1])
    pb.add_argument("--format", choices=sorted(WRITERS), default="md")
    pb.add_argument("--no-words", action="store_true")
    pb.add_argument("--force", action="store_true", help="overwrite existing outputs")
    pb.set_defaults(func=_cmd_batch)

    pf = sub.add_parser("find", help="transcribe + grep for a phrase, print timestamped hits")
    pf.add_argument("path")
    pf.add_argument("needle")
    pf.add_argument(common_model[0], **common_model[1])
    pf.add_argument(common_lang[0], **common_lang[1])
    pf.add_argument(common_cpu[0], **common_cpu[1])
    pf.set_defaults(func=_cmd_find)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
