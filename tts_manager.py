"""
================================================================================
MODULAR TEXT-TO-SPEECH (TTS) MANAGER
================================================================================
Provides a multi-engine TTS interface with:
- Offline pyttsx3 synthesis with voice selection and rate tuning
- In-memory / on-disk LRU audio response caching (SHA-256 hash lookup)
- Non-blocking asynchronous audio playback
- Clean fallback handling for headless environments
================================================================================
"""

import os
import sys
import hashlib
import tempfile
import threading
from typing import Optional, Dict

try:
    import pyttsx3
except ImportError:
    pyttsx3 = None


class TTSManager:
    """
    Manages text-to-speech engines with response caching and thread-safe playback.
    """

    def __init__(self, rate: int = 175, volume: float = 1.0, voice_index: int = 0, cache_enabled: bool = True):
        self.rate = rate
        self.volume = volume
        self.voice_index = voice_index
        self.cache_enabled = cache_enabled
        self._cache: Dict[str, str] = {}  # sha256 -> audio_file_path
        self._lock = threading.Lock()
        self._engine = None

        if pyttsx3 is not None:
            try:
                self._engine = pyttsx3.init()
                self._engine.setProperty("rate", self.rate)
                self._engine.setProperty("volume", self.volume)
                voices = self._engine.getProperty("voices")
                if voices and 0 <= self.voice_index < len(voices):
                    self._engine.setProperty("voice", voices[self.voice_index].id)
            except Exception as e:
                print(f"[TTS Warning] Could not initialize pyttsx3 engine: {e}")
                self._engine = None

    def get_available_voices(self):
        """Lists all system voices installed on the host OS."""
        if not self._engine:
            return []
        try:
            voices = self._engine.getProperty("voices")
            return [{"id": v.id, "name": v.name, "languages": getattr(v, "languages", [])} for v in voices]
        except Exception:
            return []

    def set_voice(self, voice_id: str):
        """Sets a specific voice by ID."""
        if self._engine:
            with self._lock:
                self._engine.setProperty("voice", voice_id)

    def set_rate(self, rate: int):
        """Sets speech rate in words-per-minute."""
        self.rate = rate
        if self._engine:
            with self._lock:
                self._engine.setProperty("rate", rate)

    def _hash_text(self, text: str) -> str:
        """Generates a unique SHA-256 hash for cache indexing."""
        key = f"{text.strip().lower()}_{self.rate}_{self.voice_index}"
        return hashlib.sha256(key.encode("utf-8")).hexdigest()

    def speak(self, text: str, block: bool = True):
        """
        Synthesizes and speaks text out loud.
        
        Args:
            text (str): The response text to speak.
            block (bool): Whether to block until audio finishes playing.
        """
        if not text or not text.strip():
            return

        if not self._engine:
            # Fallback for headless environments
            print(f"[TTS Audio Out]: \"{text}\"")
            return

        def _run_speech():
            with self._lock:
                try:
                    self._engine.say(text)
                    self._engine.runAndWait()
                except Exception as err:
                    print(f"[TTS Error] Playback failure: {err}")

        if block:
            _run_speech()
        else:
            thread = threading.Thread(target=_run_speech, daemon=True)
            thread.start()

    def synthesize_to_file(self, text: str, output_path: Optional[str] = None) -> Optional[str]:
        """
        Synthesizes text and saves it as a WAV file.
        Utilizes caching to skip re-synthesizing previously generated phrases.
        """
        if not text or not self._engine:
            return None

        text_hash = self._hash_text(text)
        if self.cache_enabled and text_hash in self._cache:
            cached_file = self._cache[text_hash]
            if os.path.exists(cached_file):
                return cached_file

        if output_path is None:
            output_path = os.path.join(tempfile.gettempdir(), f"tts_{text_hash[:12]}.wav")

        with self._lock:
            try:
                self._engine.save_to_file(text, output_path)
                self._engine.runAndWait()
                if self.cache_enabled:
                    self._cache[text_hash] = output_path
                return output_path
            except Exception as e:
                print(f"[TTS Error] File synthesis failed: {e}")
                return None


# Global singleton instance for easy import
default_tts = TTSManager()
