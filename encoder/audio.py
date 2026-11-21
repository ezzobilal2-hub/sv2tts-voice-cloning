from scipy.ndimage import binary_dilation
from encoder.params_data import *
from pathlib import Path
from typing import Optional, Union
from functools import lru_cache
from math import gcd
from scipy.signal import get_window, resample_poly
import numpy as np
import soundfile as sf
import struct

int16_max = (2 ** 15) - 1

def normalize_volume(wav, target_dBFS, increase_only=False, decrease_only=False):
    if increase_only and decrease_only:
        raise ValueError("Both increase only and decrease only are set")
    energy = np.mean(wav ** 2)
    if energy == 0:
        # Silent waveform: just return as-is or add tiny epsilon
        return wav + 1e-9
    dBFS_change = target_dBFS - 10 * np.log10(energy)
    return wav * (10 ** (dBFS_change / 20))


def preprocess_wav(
    fpath_or_wav: Union[str, Path, np.ndarray],
    source_sr: Optional[int] = None,
    normalize: Optional[bool] = True,
    trim_silence: Optional[bool] = True,
) -> np.ndarray:
    """
    Preprocess an audio waveform or file for the encoder:
    - Loads from disk if a path is given
    - Resamples to the encoder's sampling rate (16kHz)
    - Normalizes loudness
    - Optionally trims long silences
    - Always returns a 1-D float32 waveform
    """
    # Load the wav from disk if needed
    if isinstance(fpath_or_wav, (str, Path)):
        try:
            wav, source_sr = sf.read(str(fpath_or_wav), dtype="float32", always_2d=True)
            wav = wav.mean(axis=1)
        except (RuntimeError, OSError):
            # Older libsndfile builds need a decoder fallback for formats such as MP3.
            if not Path(fpath_or_wav).is_file():
                raise FileNotFoundError(fpath_or_wav)
            import librosa
            wav, source_sr = librosa.load(str(fpath_or_wav), sr=None, mono=True)
    else:
        wav = np.asarray(fpath_or_wav)
        if source_sr is None:
            raise ValueError("If passing a waveform array, you must provide source_sr.")

    if wav.size == 0:
        return np.empty(0, dtype=np.float32)

    # Resample the wav if needed
    if source_sr is not None and source_sr != sampling_rate:
        divisor = gcd(source_sr, sampling_rate)
        wav = resample_poly(wav, sampling_rate // divisor, source_sr // divisor)

    # Apply the preprocessing: normalize volume and shorten long silences
    if normalize:
        wav = normalize_volume(wav, audio_norm_target_dBFS, increase_only=True)
    if trim_silence:
        energy = wav ** 2
        frames = np.nonzero(energy > 1e-6)[0]
        if frames.size > 0:
            wav = wav[frames[0]:frames[-1] + 1]

    # ✅ Ensure waveform is float32 and 1-D
    return wav.astype(np.float32).flatten()



@lru_cache(maxsize=1)
def _encoder_filters():
    """Periodic Hann window and area-normalized Slaney mel filters for the encoder."""
    n_fft = sampling_rate * mel_window_length // 1000
    nyquist = sampling_rate / 2
    # Slaney's scale is linear below 1 kHz and logarithmic above it.
    log_step = np.log(6.4) / 27
    max_mel = 15 + np.log(nyquist / 1000) / log_step
    mel = np.linspace(0, max_mel, mel_n_channels + 2)
    edges = np.where(mel < 15, mel * (1000 / 15), 1000 * np.exp((mel - 15) * log_step))
    widths = np.diff(edges)
    ramps = edges[:, None] - np.fft.rfftfreq(n_fft, 1 / sampling_rate)
    triangles = np.minimum(-ramps[:-2] / widths[:-1, None], ramps[2:] / widths[1:, None])
    filters = np.maximum(0, triangles).astype(np.float32)
    filters *= (2 / (edges[2:] - edges[:-2]))[:, None]
    return get_window("hann", n_fft), filters


def wav_to_mel_spectrogram(wav):
    window, filters = _encoder_filters()
    hop = sampling_rate * mel_window_step // 1000
    if len(wav) < len(window):
        wav = np.pad(wav, (0, len(window) - len(wav)))
    frames = np.lib.stride_tricks.sliding_window_view(wav, len(window))[::hop]
    spectrum = np.fft.rfft(frames * window).astype(np.complex64)
    power = filters @ (np.abs(spectrum).T ** 2)
    db = 10 * np.log10(np.maximum(power, 1e-10))
    return np.maximum(db - db.max(), -80).T



def trim_long_silences(wav):
    """
    Ensures that segments without voice in the waveform remain no longer than a 
    threshold determined by the VAD parameters in params.py.

    :param wav: the raw waveform as a numpy array of floats 
    :return: the same waveform with silences trimmed away (length <= original wav length)
    """
    # Compute the voice detection window size
    samples_per_window = (vad_window_length * sampling_rate) // 1000
    
    # Trim the end of the audio to have a multiple of the window size
    wav = wav[:len(wav) - (len(wav) % samples_per_window)]
    
    # Convert the float waveform to 16-bit mono PCM
    pcm_wave = struct.pack("%dh" % len(wav), *(np.round(wav * int16_max)).astype(np.int16))
    
    # Perform voice activation detection
    voice_flags = []
    import webrtcvad
    vad = webrtcvad.Vad(mode=3)
    for window_start in range(0, len(wav), samples_per_window):
        window_end = window_start + samples_per_window
        voice_flags.append(vad.is_speech(pcm_wave[window_start * 2:window_end * 2],
                                         sample_rate=sampling_rate))
    voice_flags = np.array(voice_flags)
    
    # Smooth the voice detection with a moving average
    def moving_average(array, width):
        array_padded = np.concatenate((np.zeros((width - 1) // 2), array, np.zeros(width // 2)))
        ret = np.cumsum(array_padded, dtype=float)
        ret[width:] = ret[width:] - ret[:-width]
        return ret[width - 1:] / width
    
    audio_mask = moving_average(voice_flags, vad_moving_average_width)
    audio_mask = np.round(audio_mask).astype(bool)
    
    # Dilate the voiced regions
    audio_mask = binary_dilation(audio_mask, np.ones(vad_max_silence_length + 1))
    audio_mask = np.repeat(audio_mask, samples_per_window)
    
    return wav[audio_mask == True]




