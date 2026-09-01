"""
================================================================================
DSP NOISE REDUCTION BENCHMARKING SCRIPT
================================================================================
Benchmarks the Butterworth bandpass filter and STFT spectral subtraction
against varying noise profiles (AC mains hum, white noise, high-frequency hiss).
Calculates Signal-to-Noise Ratio (SNR) improvements and spectral centroid shifts.
================================================================================
"""

import sys
import os
import numpy as np

# Ensure UTF-8 output on Windows
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from audio_dsp import (
    apply_bandpass_filter,
    spectral_subtraction,
    preprocess_speech_audio,
    compute_snr,
    compute_spectral_centroid
)


def run_benchmark():
    print("=" * 70)
    print(" [DSP] AUDIO ENHANCEMENT & NOISE REDUCTION BENCHMARK")
    print("=" * 70)

    fs = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)

    # 1. Generate Clean Speech Surrogate (Multi-formant vocal harmonic synthetic signal)
    f0 = 150.0  # Fundamental frequency (pitch)
    speech = (
        0.45 * np.sin(2 * np.pi * f0 * t) +
        0.30 * np.sin(2 * np.pi * (2 * f0) * t) +
        0.25 * np.sin(2 * np.pi * 800 * t) +    # First vowel formant ~800 Hz
        0.20 * np.sin(2 * np.pi * 1800 * t) +   # Second vowel formant ~1800 Hz
        0.15 * np.sin(2 * np.pi * 2500 * t)     # Third vowel formant ~2500 Hz
    ).astype(np.float32)

    # 2. Generate Noise Profiles
    hum_60hz = 0.35 * np.sin(2 * np.pi * 60 * t)      # AC hum
    hiss_6khz = 0.25 * np.sin(2 * np.pi * 6500 * t)   # High-freq mic hiss
    white_noise = 0.15 * np.random.normal(0, 1, len(t)) # Ambient broadband noise
    dc_bias = 0.18

    corrupted = speech + hum_60hz + hiss_6khz + white_noise + dc_bias

    # Compute Initial Metrics
    init_snr = compute_snr(speech, corrupted)
    init_centroid = compute_spectral_centroid(corrupted, fs=fs)
    clean_centroid = compute_spectral_centroid(speech, fs=fs)

    print("\nBaseline Metrics:")
    print(f"   - Clean Signal Spectral Centroid : {clean_centroid:.2f} Hz")
    print(f"   - Corrupted Signal SNR           : {init_snr:.2f} dB")
    print(f"   - Corrupted Spectral Centroid    : {init_centroid:.2f} Hz")

    # Step A: Butterworth Bandpass (300Hz - 3400Hz)
    bp_filtered = apply_bandpass_filter(corrupted - dc_bias, lowcut=300.0, highcut=3400.0, fs=fs, order=4)
    bp_snr = compute_snr(speech, bp_filtered)
    bp_centroid = compute_spectral_centroid(bp_filtered, fs=fs)

    print("\nStage 1: 4th-Order Butterworth Bandpass (300Hz - 3400Hz):")
    print(f"   - Enhanced SNR                   : {bp_snr:.2f} dB (+{bp_snr - init_snr:.2f} dB improvement)")
    print(f"   - Post-Filter Spectral Centroid  : {bp_centroid:.2f} Hz")

    # Step B: Full DSP Pipeline (Bandpass + Spectral Subtraction + Normalization)
    full_enhanced = preprocess_speech_audio(corrupted, fs=fs, apply_filter=True, apply_spectral_sub=True)
    full_snr = compute_snr(speech, full_enhanced)
    full_centroid = compute_spectral_centroid(full_enhanced, fs=fs)

    print("\nStage 2: Full DSP Preprocessing Pipeline (Bandpass + Spectral Subtraction):")
    print(f"   - Enhanced SNR                   : {full_snr:.2f} dB (+{full_snr - init_snr:.2f} dB net improvement)")
    print(f"   - Final Spectral Centroid        : {full_centroid:.2f} Hz")
    print(f"   - Peak Normalization Range       : [{np.min(full_enhanced):.2f}, {np.max(full_enhanced):.2f}]")

    print("\n" + "=" * 70)
    print(f"[SUCCESS] DSP Benchmark Completed: Net SNR Gain of +{full_snr - init_snr:.2f} dB achieved.")
    print("=" * 70)


if __name__ == "__main__":
    run_benchmark()
