"""
Unit tests for Text-to-Speech Manager and caching layer.
"""

import os
import sys

# Ensure parent directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tts_manager import TTSManager


def test_tts_manager_initialization():
    tts = TTSManager(rate=180, volume=0.9, cache_enabled=True)
    assert tts.rate == 180
    assert tts.volume == 0.9
    assert tts.cache_enabled is True


def test_tts_hash_generation():
    tts = TTSManager(rate=175)
    h1 = tts._hash_text("Hello world")
    h2 = tts._hash_text("hello world")
    assert h1 == h2  # Case insensitive match

    h3 = tts._hash_text("Different text")
    assert h1 != h3


def test_tts_fallback_speak():
    tts = TTSManager()
    tts.speak("Test voice synthesis fallback", block=True)
    tts.speak("", block=True)  # Empty string handling
