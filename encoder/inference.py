from encoder.model import SpeakerEncoder
from encoder.audio import wav_to_mel_spectrogram, preprocess_wav as audio_preprocess_wav
from encoder.params_model import encoder_sampling_rate
import numpy as np
import torch

_model = None
_device = None
sampling_rate = encoder_sampling_rate

def load_model(weights_fpath: str, device=None):
    global _model, _device
    if device is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        _device = torch.device(device)

    _model = SpeakerEncoder(_device, torch.device("cpu"))
    checkpoint = torch.load(weights_fpath, map_location=_device)
    _model.load_state_dict(checkpoint["model_state"])
    _model.eval()
    print(f'Loaded encoder "{weights_fpath}" trained to step {checkpoint["step"]}')

def is_loaded() -> bool:
    return _model is not None

@torch.no_grad()
def embed_utterance(wav: np.ndarray):
    if _model is None:
        raise Exception("Model not loaded. Call load_model() first.")
    if wav.dtype != np.float32:
        wav = wav.astype(np.float32)
    frames = wav_to_mel_spectrogram(wav)
    frames_torch = torch.from_numpy(frames[None, ...]).float().to(_device)
    embed = _model.forward(frames_torch).detach().cpu().numpy()[0]
    return embed

# ------------------- Fix preprocess wrapper -------------------
def preprocess_wav(wav, source_sr: int = 16000, normalize=True, trim_silence=True) -> np.ndarray:
    return audio_preprocess_wav(wav, source_sr, normalize, trim_silence)


def embed_reference(path):
    wav = preprocess_wav(path)
    if not wav.size or not np.isfinite(wav).all() or np.max(np.abs(wav)) < 1e-6:
        raise ValueError("The reference must contain audible samples.")
    return embed_utterance(wav)

# ------------------------------ Plotting ------------------------------
def plot_embedding_as_heatmap(embedding: np.ndarray, ax=None, title="Speaker Embedding"):
    import matplotlib.pyplot as plt
    if embedding.ndim == 1:
        embedding = embedding[None, :]

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 2))

    im = ax.imshow(embedding, aspect='auto', origin='lower', cmap='viridis')
    ax.set_title(title)
    ax.set_xlabel("Embedding Dimension")
    ax.set_ylabel("Frame" if embedding.shape[0] > 1 else "Embedding")
    plt.colorbar(im, ax=ax)
    ax.figure.tight_layout()
