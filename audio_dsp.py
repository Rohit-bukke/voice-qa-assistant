"""
================================================================================
DSP AUDIO PREPROCESSING MODULE
================================================================================
Advanced digital signal processing pipeline for real-time speech enhancement:
1. 4th-order Butterworth Bandpass Filter (300 Hz - 3400 Hz) via SOS format
2. Short-Time Fourier Transform (STFT) Spectral Subtraction & Noise Gating
3. Dynamic Hardware DC Bias Removal & Peak Amplitude Normalization
4. Signal-to-Noise Ratio (SNR) and Spectral Centroid Measurement
================================================================================
"""

import numpy as np
from scipy.signal import butter, sosfiltfilt, stft, istft


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


def spectral_subtraction(audio_data, fs=16000, nperseg=512, noise_frames=6, oversubtraction=1.0, spectral_floor=0.05):
    """
    Applies STFT-based spectral subtraction to suppress stationary background noise.
    
    Args:
        audio_data (ndarray): Input 1D audio array.
        fs (int): Sampling frequency.
        nperseg (int): Segment length for STFT.
        noise_frames (int): Number of initial frames assumed to be background noise.
        oversubtraction (float): Factor alpha to over-subtract noise magnitude.
        spectral_floor (float): Minimum residual spectral magnitude factor beta.
        
    Returns:
        ndarray: Denoised audio signal.
    """
    if len(audio_data) < nperseg * 2:
        return audio_data

    # 1. Compute Short-Time Fourier Transform
    f, t, Zxx = stft(audio_data, fs=fs, nperseg=nperseg, noverlap=nperseg // 2)
    magnitude = np.abs(Zxx)
    phase = np.angle(Zxx)

    # 2. Estimate average noise power spectrum from initial non-speech frames
    n_frames = min(noise_frames, magnitude.shape[1])
    noise_estimate = np.mean(magnitude[:, :n_frames], axis=1, keepdims=True)

    # 3. Spectral Subtraction with spectral floor
    subtracted_mag = np.maximum(magnitude - (oversubtraction * noise_estimate), spectral_floor * magnitude)

    # 4. Reconstruct complex spectrum with original phase
    Zxx_clean = subtracted_mag * np.exp(1j * phase)

    # 5. Inverse STFT to return to time domain
    _, clean_audio = istft(Zxx_clean, fs=fs, nperseg=nperseg, noverlap=nperseg // 2)
    
    # Trim or pad to match original length
    if len(clean_audio) > len(audio_data):
        clean_audio = clean_audio[:len(audio_data)]
    elif len(clean_audio) < len(audio_data):
        clean_audio = np.pad(clean_audio, (0, len(audio_data) - len(clean_audio)))

    return clean_audio.astype(np.float32)


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


def compute_spectral_centroid(audio_data, fs=16000):
    """
    Computes the spectral centroid (center of mass of spectrum) in Hz.
    Useful for characterizing brightness / frequency concentration of the audio.
    """
    if len(audio_data) < 64:
        return 0.0
    fft_vals = np.abs(np.fft.rfft(audio_data))
    freqs = np.fft.rfftfreq(len(audio_data), 1.0 / fs)
    total_energy = np.sum(fft_vals)
    if total_energy < 1e-6:
        return 0.0
    return float(np.sum(freqs * fft_vals) / total_energy)


def preprocess_speech_audio(audio_data, fs=16000, apply_filter=True, apply_spectral_sub=False):
    """
    Full DSP Preprocessing Pipeline for Voice Assistant Input:
    1. Flatten to 1D mono float32 array
    2. Remove hardware DC offset
    3. Apply Butterworth Bandpass Filter (300Hz-3400Hz)
    4. Optional Spectral Subtraction for stationarity noise
    5. Peak amplitude normalization
    
    Args:
        audio_data (ndarray): Input audio array (float32, range [-1.0, 1.0])
        fs (int): Audio sample rate in Hz
        apply_filter (bool): Whether to apply Butterworth filtering
        apply_spectral_sub (bool): Whether to apply STFT spectral subtraction
        
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
        
    # Step 3: Optional Spectral Subtraction
    if apply_spectral_sub and len(audio) > 1024:
        audio = spectral_subtraction(audio, fs=fs)

    # Step 4: Peak normalization
    audio = normalize_audio(audio, target_peak=0.95)
    
    return audio
