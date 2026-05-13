"""Model selection + cache location for faster-whisper.

We pin the HF cache to `/mnt/d/cache/huggingface/` so model weights (often
1-2 GB each) don't bloat the WSL VHD on C:.

`pick_compute_type()` chooses float16 if CUDA is available, int8 otherwise.
"""

from __future__ import annotations

import os
from pathlib import Path

CACHE_ROOT = Path("/mnt/d/cache/huggingface")
DEFAULT_MODEL = "distil-large-v3"

_ALIAS = {
    "tiny": "tiny",
    "base": "base",
    "small": "small",
    "medium": "medium",
    "large": "large-v3",
    "large-v3": "large-v3",
    "distil": "distil-large-v3",
    "distil-large-v3": "distil-large-v3",
}


def configure_cache() -> None:
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(CACHE_ROOT))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(CACHE_ROOT / "hub"))


def resolve_model(name: str) -> str:
    return _ALIAS.get(name, name)


def pick_compute_type(prefer_gpu: bool = True) -> tuple[str, str]:
    """Return (device, compute_type). CUDA → ('cuda','float16'); else ('cpu','int8')."""
    if prefer_gpu:
        try:
            import ctranslate2  # noqa: F401
            import subprocess
            r = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=3)
            if r.returncode == 0 and "GPU" in r.stdout:
                return "cuda", "float16"
        except Exception:
            pass
    return "cpu", "int8"
