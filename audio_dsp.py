"""
================================================================================
DSP AUDIO PREPROCESSING MODULE
================================================================================
Applies digital signal processing techniques to enhance microphone speech
prior to feeding it into Whisper STT.

Integrates concepts from DSP-based-Speech-Noise-Reduction:
1. 4th-order Butterworth Bandpass Filter (300 Hz - 3400 Hz) to isolate human voice
2. DC offset removal and peak normalization
3. Signal-to-Noise Ratio (SNR) measurement and improvement validation
================================================================================
"""

import numpy as np
from scipy.signal import butter, sosfiltfilt


def butter_bandpass(lowcut=300.0, highcut=3400.0, fs=16000, order=4):
    """
    Designs a Butterworth bandpass filter in Second-Order Sections (SOS) format
    for superior numerical stability.
    
    Args:
        lowcut (float): Lower cutoff frequency in Hz (standard speech low cutoff: ~300 Hz)
        highcut (float): Upper cutoff frequency in Hz (standard speech high cutoff: ~3400 Hz)
        fs (int): Sampling rate in Hz (default 16000 Hz for Whisper)
        order (int): Filter order (4th order provides sharp roll-off without ringing)
        
    Returns:
        ndarray: Second-order sections representation of the IIR filter.
    """
    nyq = 0.5 * fs
    low = max(lowcut / nyq, 0.001)
    high = min(highcut / nyq, 0.999)
    sos = butter(order, [low, high], btype='bandpass', output='sos')
    return sos


def apply_bandpass_filter(audio_data, lowcut=300.0, highcut=3400.0, fs=16000, order=4):
    """
    Applies zero-phase forward-backward Butterworth bandpass filtering.
    Zero-phase filtering preserves time alignment and avoids phase distortion.
    """
    if len(audio_data) < 30:
        return audio_data
    
    sos = butter_bandpass(lowcut, highcut, fs, order=order)
    # sosfiltfilt applies zero-phase forward-backward filtering
    filtered = sosfiltfilt(sos, audio_data)
    return filtered.astype(np.float32)


def remove_dc_offset(audio_data):
    """
    Removes DC bias (mean offset) caused by microphone hardware.
    """
    return audio_data - np.mean(audio_data)


def normalize_audio(audio_data, target_peak=0.95):
    """
    Normalizes audio peak amplitude to prevent clipping and ensure consistent levels.
    """
    max_val = np.max(np.abs(audio_data))
    if max_val > 1e-6:
        return (audio_data / max_val) * target_peak
    return audio_data


def compute_snr(signal_clean, signal_noisy):
    """
    Computes Signal-to-Noise Ratio (SNR) in decibels (dB).
    SNR = 10 * log10(Power_signal / Power_noise)
    """
    noise = signal_noisy - signal_clean
    power_signal = np.mean(signal_clean ** 2)
    power_noise = np.mean(noise ** 2)
    
    if power_noise < 1e-12:
        return float("inf")
    if power_signal < 1e-12:
        return float("-inf")
        
    snr = 10 * np.log10(power_signal / power_noise)
    return float(snr)


def preprocess_speech_audio(audio_data, fs=16000, apply_filter=True):
    """
    Full DSP Preprocessing Pipeline for Voice Assistant Input:
    1. Flatten to 1D mono float32 array
    2. Remove hardware DC offset
    3. Apply Butterworth Bandpass Filter (300Hz-3400Hz)
    4. Peak amplitude normalization
    
    Args:
        audio_data (ndarray): Input audio array (float32, range [-1.0, 1.0])
        fs (int): Audio sample rate in Hz
        apply_filter (bool): Whether to apply Butterworth filtering
        
    Returns:
        ndarray: Enhanced, filtered float32 audio array ready for Whisper STT.
    """
    audio = np.asarray(audio_data, dtype=np.float32).flatten()
    
    if len(audio) == 0:
        return audio
        
    # Step 1: Remove DC bias
    audio = remove_dc_offset(audio)
    
    # Step 2: Apply bandpass filter to suppress low hums (<300Hz) & high hiss (>3400Hz)
    if apply_filter and len(audio) > 64:
        audio = apply_bandpass_filter(audio, lowcut=300.0, highcut=3400.0, fs=fs, order=4)
        
    # Step 3: Peak normalization
    audio = normalize_audio(audio, target_peak=0.95)
    
    return audio
