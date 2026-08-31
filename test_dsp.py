"""
Test suite to verify DSP filter math, audio preprocessing, and system modules.
"""

import numpy as np
from audio_dsp import (
    butter_bandpass,
    apply_bandpass_filter,
    remove_dc_offset,
    normalize_audio,
    compute_snr,
    preprocess_speech_audio
)


def test_dsp_pipeline():
    print("Testing DSP Pipeline components...")
    fs = 16000
    t = np.linspace(0, 1.0, fs, endpoint=False)

    # Synthetic signal: 1000 Hz human voice formant + 60 Hz hum + 6000 Hz hiss + DC bias
    speech_signal = 0.5 * np.sin(2 * np.pi * 1000 * t)
    noise_hum = 0.3 * np.sin(2 * np.pi * 60 * t)
    noise_hiss = 0.2 * np.sin(2 * np.pi * 6000 * t)
    dc_bias = 0.15
    noisy_signal = speech_signal + noise_hum + noise_hiss + dc_bias

    # 1. Test DC offset removal
    no_dc = remove_dc_offset(noisy_signal)
    assert abs(np.mean(no_dc)) < 1e-6, "DC offset removal failed"
    print(" DC offset removal passed.")

    # 2. Test Butterworth Bandpass filtering
    filtered = apply_bandpass_filter(noisy_signal, lowcut=300.0, highcut=3400.0, fs=fs, order=4)
    assert len(filtered) == len(noisy_signal), "Filtered audio length mismatch"
    print(" Butterworth Bandpass filter execution passed.")

    # 3. Test Full Preprocessing
    preprocessed = preprocess_speech_audio(noisy_signal, fs=fs, apply_filter=True)
    assert np.max(np.abs(preprocessed)) <= 0.96, "Normalization peak exceeded"
    print(" Preprocess audio pipeline passed.")

    # 4. Test SNR computation
    initial_snr = compute_snr(speech_signal, noisy_signal)
    filtered_speech = apply_bandpass_filter(noisy_signal - dc_bias, lowcut=300.0, highcut=3400.0, fs=fs, order=4)
    enhanced_snr = compute_snr(speech_signal, filtered_speech)
    print(f" Initial SNR: {initial_snr:.2f} dB | Enhanced SNR after Bandpass: {enhanced_snr:.2f} dB")
    assert enhanced_snr > initial_snr, "Filtered signal should have higher SNR"
    print(" SNR improvement validation passed.")

    print("\n All DSP verification tests passed successfully!")


if __name__ == "__main__":
    test_dsp_pipeline()
