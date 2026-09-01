"""
Standard library test runner for environment compatibility.
"""

import unittest
import sys
import os

# Add parent directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tests.test_dsp import (
    test_dc_offset_removal,
    test_normalize_audio,
    test_butterworth_bandpass_stability,
    test_bandpass_filtering_snr_gain,
    test_preprocess_speech_audio_pipeline,
    test_spectral_centroid_calculation
)
from tests.test_memory import (
    test_turn_creation,
    test_sliding_window_turn_pruning,
    test_sliding_window_token_budgeting,
    test_session_manager_persistence
)
from tests.test_tts import (
    test_tts_manager_initialization,
    test_tts_hash_generation,
    test_tts_fallback_speak
)


class VoiceAssistantTestSuite(unittest.TestCase):
    def test_all_dsp_functions(self):
        test_dc_offset_removal()
        test_normalize_audio()
        test_butterworth_bandpass_stability()
        test_bandpass_filtering_snr_gain()
        test_preprocess_speech_audio_pipeline()
        test_spectral_centroid_calculation()

    def test_all_memory_functions(self):
        test_turn_creation()
        test_sliding_window_turn_pruning()
        test_sliding_window_token_budgeting()
        test_session_manager_persistence()

    def test_all_tts_functions(self):
        test_tts_manager_initialization()
        test_tts_hash_generation()
        test_tts_fallback_speak()


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(VoiceAssistantTestSuite)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
