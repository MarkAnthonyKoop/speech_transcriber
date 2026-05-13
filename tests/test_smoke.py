"""Smoke test for speech_transcriber — verifies the package imports cleanly."""
import importlib


def test_import():
    mod = importlib.import_module("speech_transcriber")
    assert mod is not None


def test_main_module_importable():
    # `python3 -m speech_transcriber` works iff this import works
    importlib.import_module("speech_transcriber.__main__")
