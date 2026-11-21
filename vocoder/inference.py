from vocoder.models.fatchord_version import WaveRNN
from vocoder import hparams as hp
import torch


_model = None   # type: WaveRNN

def load_model(weights_fpath, verbose=True):
    global _model, _device
    
    if verbose:
        print("Building Wave-RNN")
    _model = WaveRNN(
        rnn_dims=hp.voc_rnn_dims,
        fc_dims=hp.voc_fc_dims,
        bits=hp.bits,
        pad=hp.voc_pad,
        upsample_factors=hp.voc_upsample_factors,
        feat_dims=hp.num_mels,
        compute_dims=hp.voc_compute_dims,
        res_out_dims=hp.voc_res_out_dims,
        res_blocks=hp.voc_res_blocks,
        hop_length=hp.hop_length,
        sample_rate=hp.sample_rate,
        mode=hp.voc_mode
    )

    if torch.cuda.is_available():
        _model = _model.cuda()
        _device = torch.device('cuda')
    else:
        _device = torch.device('cpu')
    
    if verbose:
        print("Loading model weights at %s" % weights_fpath)
    checkpoint = torch.load(weights_fpath, _device)
    _model.load_state_dict(checkpoint['model_state'])
    _model.eval()


def is_loaded():
    return _model is not None


def infer_waveform(mel, normalize=True, batched=True, target=None, overlap=None,
                   progress_callback=None, *, fast=False):
    """Generate a waveform from a synthesizer mel spectrogram.

    Fast mode uses shorter overlapping chunks for parallel generation. Set
    target or overlap explicitly to bypass automatic CPU chunk selection.
    With one explicit setting, the other defaults to an overlap of 800 or
    a target of 1600 (fast) or 8000 (standard), in waveform samples.
    """
    if _model is None:
        raise Exception("Please load Wave-RNN in memory before using it")
    if fast and batched and target is None and overlap is None and _device.type == "cpu":
        # Smaller windows finish short clips sooner. Limit this preset to 32
        # parallel chunks so longer clips do not double the recurrent batch size.
        short_target, short_overlap = 800, 400
        max_samples = 32 * (short_target + short_overlap) + short_overlap
        if mel.shape[-1] * _model.hop_length <= max_samples:
            target, overlap = short_target, short_overlap
    if target is None:
        target = 1600 if fast else 8000
    if overlap is None:
        overlap = 800
    if batched and (target < 1 or overlap < 1):
        raise ValueError("Batched generation requires positive target and overlap lengths.")
    
    if normalize:
        mel = mel / hp.mel_max_abs_value
    mel = torch.from_numpy(mel[None, ...])
    wav = _model.generate(mel, batched, target, overlap, hp.mu_law, progress_callback)
    return wav
