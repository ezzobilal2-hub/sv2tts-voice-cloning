# Real-Time Voice Cloning
This bachelor's graduation project adapts Corentin Jemine's Real-Time Voice Cloning implementation of SV2TTS. It generates English speech from new text and a short reference recording. This version includes a desktop interface, a command-line demo, recording-pair evaluation, and an optional fast CPU mode
SV2TTS is a deep learning framework in three stages. In the first stage, one creates a digital representation of a voice from a few seconds of audio. In the second and third stages, this representation is used as reference to generate speech given arbitrary text.

## Setup

### Install

Use a Python 3.10 virtual environment. Install PyTorch for your CPU or CUDA device, then run `python -m pip install -r requirements.txt`. The demos use the encoder, synthesizer, and vocoder in `saved_models/default/`. Missing default checkpoints are downloaded on first use.

The first demo run needs internet access to download about 441 MB of model weights. For offline use, place `encoder.pt`, `synthesizer.pt`, and `vocoder.pt` in `saved_models/default/` before starting. The CPU demo has been exercised with Python 3.10 and PyTorch 2.7.0. [PyTorch installer](https://pytorch.org/get-started/locally/) · [Pretrained checkpoints](https://huggingface.co/CorentinJ/SV2TTS/tree/main)

### Generate speech

Save about five seconds of your own speech as `reference.wav` in the project folder. Run the following command there to generate `voice-test.wav`.

```shell
python demo_cli.py --cpu --no-sound --fast --reference reference.wav --text "This is a voice test." --output voice-test.wav
```

The example uses `--fast` for shorter overlapping WaveRNN chunks. On CPU, clips up to about 2.4 seconds use smaller chunks to reduce latency. Chunking can change how the voice sounds at the joins. CPU generation took longer than playback in the short-clip benchmarks. Add `--seed 42` to repeat a run with the same model, device, and settings. Run `python demo_cli.py` for interactive input. Press Enter at the reference prompt to exit.

### Use the GUI

Run `python demo_toolbox.py`. Record five seconds with the system’s default microphone or open an existing reference WAV. Enter a sentence and choose **Synthesize**. The app saves the result under `generated/`. Clear **Play generated audio** to use the GUI without an output device.

### Evaluate recording pairs

Run `python Results.py` to score local pairs such as `recordings/recorded_1234567890.wav` and `generated/cloned_1234567890.wav`. The GUI creates this naming pattern when you record with the microphone. The evaluator writes `evaluation_results.csv` in the current directory. The cosine score measures speaker-embedding similarity. The `mos_estimate` column is a heuristic derived from automatic metrics. A listening study is needed to report a human mean opinion score.

For training data, see the [supported datasets and folder layouts](https://github.com/CorentinJ/Real-Time-Voice-Cloning/wiki/Training#datasets).
