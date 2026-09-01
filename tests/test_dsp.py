"""
Unit tests for DSP audio preprocessing pipeline and filters.
Compatible with pytest and standard python unittest.
"""

import sys
import os
import numpy as np

# Ensure parent directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from audio_dsp import (
    butter_bandpass,
    apply_bandpass_filter,
    spectral_subtraction,
    remove_dc_offset,
    normalize_audio,
    compute_snr,
    compute_spectral_centroid,
    preprocess_speech_audio
)


def test_dc_offset_removal():
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    cleaned = remove_dc_offset(data)
    assert abs(np.mean(cleaned)) < 1e-6


def test_normalize_audio():
    data = np.array([-2.0, 0.5, 3.0], dtype=np.float32)
    normalized = normalize_audio(data, target_peak=0.95)
    assert abs(np.max(np.abs(normalized)) - 0.95) < 1e-3


def test_butterworth_bandpass_stability():
    fs = 16000
    sos = butter_bandpass(lowcut=300.0, highcut=3400.0, fs=fs, order=4)
    # Bandpass of order 4 has 4 second-order sections (shape 4x6)
    assert sos.shape == (4, 6), f"Expected shape (4, 6), got {sos.shape}"


def test_bandpass_filtering_snr_gain():
    fs = 16000
    t = np.linspace(0, 1.0, fs, endpoint=False)
    
    # 1000 Hz clean speech + 60 Hz hum + 6000 Hz hiss + DC
    clean = 0.5 * np.sin(2 * np.pi * 1000 * t)
    noisy = clean + 0.3 * np.sin(2 * np.pi * 60 * t) + 0.2 * np.sin(2 * np.pi * 6000 * t) + 0.1
    
    initial_snr = compute_snr(clean, noisy)
    filtered = apply_bandpass_filter(noisy - 0.1, lowcut=300.0, highcut=3400.0, fs=fs, order=4)
    enhanced_snr = compute_snr(clean, filtered)
    
    assert enhanced_snr > initial_snr
    assert enhanced_snr - initial_snr > 10.0  # Suppresses out-of-band noise by > 10 dB


def test_preprocess_speech_audio_pipeline():
    fs = 16000
    t = np.linspace(0, 0.5, int(fs * 0.5), endpoint=False)
    signal = 0.4 * np.sin(2 * np.pi * 500 * t) + 0.2
    
    processed = preprocess_speech_audio(signal, fs=fs, apply_filter=True, apply_spectral_sub=False)
    assert len(processed) == len(signal)
    assert np.max(np.abs(processed)) <= 0.96
    assert abs(np.mean(processed)) < 0.05


def test_spectral_centroid_calculation():
    fs = 16000
    t = np.linspace(0, 1.0, fs, endpoint=False)
    pure_1khz = np.sin(2 * np.pi * 1000 * t)
    centroid = compute_spectral_centroid(pure_1khz, fs=fs)
    assert abs(centroid - 1000.0) < 50.0  # Centroid of 1kHz tone should be ~1000 Hz
