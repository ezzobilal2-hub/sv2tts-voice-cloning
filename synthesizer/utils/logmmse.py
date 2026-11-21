import numpy as np
from scipy.signal import stft, istft

def profile_noise(noisy_signal: np.ndarray, sample_rate: int):
    """
    Simple noise profile: takes the first 0.5 seconds of audio as noise
    and returns its STFT magnitude.
    """
    n_fft = 512
    hop_length = n_fft // 2
    # Take first 0.5 sec as noise
    noise_sample = noisy_signal[:int(0.5 * sample_rate)]
    f, t, Zxx = stft(noise_sample, fs=sample_rate, nperseg=n_fft, noverlap=hop_length)
    noise_mag = np.abs(Zxx)
    return noise_mag

def denoise(noisy_signal: np.ndarray, noise_profile: np.ndarray, eta=0):
    """
    Simple LogMMSE denoising: subtract the noise magnitude from the signal STFT.
    """
    n_fft = 512
    hop_length = n_fft // 2
    f, t, Zxx = stft(noisy_signal, nperseg=n_fft, noverlap=hop_length)
    magnitude = np.abs(Zxx)
    phase = np.angle(Zxx)

    # Subtract noise magnitude with floor at 0
    denoised_mag = np.maximum(magnitude - noise_profile.mean(axis=1, keepdims=True), 0.0)

    # Reconstruct waveform
    Zxx_denoised = denoised_mag * np.exp(1j * phase)
    _, denoised_signal = istft(Zxx_denoised, nperseg=n_fft, noverlap=hop_length)
    return denoised_signal.astype(np.float32)
