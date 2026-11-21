import argparse
import os
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from encoder import inference as encoder
from synthesizer.inference import Synthesizer
from utils.audio_device import play_wav
from utils.default_models import MODELS_DIR, ensure_default_models
from vocoder import inference as vocoder


def generate(reference, text, output, synthesizer, fast):
    if not reference.is_file():
        raise FileNotFoundError(reference)
    if not text.strip():
        raise ValueError("Enter a sentence to synthesize.")
    embedding = encoder.embed_reference(reference)
    mel = synthesizer.synthesize_spectrograms([text], [embedding])[0]
    wave = vocoder.infer_waveform(mel, fast=fast)
    # Use the encoder's original output normalization and silence trimming.
    wave = encoder.preprocess_wav(wave, source_sr=synthesizer.sample_rate)
    sf.write(str(output), wave, synthesizer.sample_rate)
    print(f"Saved {output} ({len(wave) / synthesizer.sample_rate:.2f} s)")


def main():
    parser = argparse.ArgumentParser(description="Clone a reference voice from a recording and text.")
    default_dir = MODELS_DIR / "default"
    parser.add_argument("-e", "--enc_model_fpath", type=Path, default=default_dir / "encoder.pt")
    parser.add_argument("-s", "--syn_model_fpath", type=Path, default=default_dir / "synthesizer.pt")
    parser.add_argument("-v", "--voc_model_fpath", type=Path, default=default_dir / "vocoder.pt")
    parser.add_argument("--cpu", action="store_true", help="Use the CPU even if CUDA is available.")
    parser.add_argument("--no_sound", "--no-sound", action="store_true", help="Save without playback.")
    parser.add_argument("--seed", type=int, help="Random seed for repeatable generation.")
    parser.add_argument("--fast", action="store_true", help="Use shorter overlapping vocoder chunks.")
    parser.add_argument("--reference", type=Path, help="Reference recording for one-shot generation.")
    parser.add_argument("--text", help="Sentence for one-shot generation.")
    parser.add_argument("--output", type=Path, help="WAV destination for one-shot generation.")
    args = parser.parse_args()
    one_shot = args.reference is not None
    if one_shot != (args.text is not None) or (args.output is not None and not one_shot):
        parser.error("Use --reference and --text together. --output is optional.")

    if args.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    if not torch.cuda.is_available():
        torch.set_num_threads(min(torch.get_num_threads(), 3))
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    paths = dict(encoder=args.enc_model_fpath, synthesizer=args.syn_model_fpath, vocoder=args.voc_model_fpath)
    defaults = [name for name, path in paths.items() if path.resolve() == default_dir / f"{name}.pt"]
    ensure_default_models(MODELS_DIR, defaults)
    encoder.load_model(args.enc_model_fpath)
    synthesizer = Synthesizer(args.syn_model_fpath)
    vocoder.load_model(args.voc_model_fpath)

    index = 0
    while True:
        try:
            if one_shot:
                reference, text = args.reference, args.text
            else:
                name = input("Reference WAV path (Enter to quit): ").strip().strip('"')
                if not name:
                    return
                reference = Path(name)
                text = input("Sentence to synthesize: ")
            output = args.output
            if output is None:
                output = Path(f"demo_output_{index:02d}.wav")
                while output.exists():
                    index += 1
                    output = Path(f"demo_output_{index:02d}.wav")
            generate(reference, text, output, synthesizer, args.fast)
            if not args.no_sound:
                try:
                    play_wav(output, blocking=one_shot)
                except Exception as exc:
                    print(f"Playback unavailable: {exc}")
            if one_shot:
                return
        except (EOFError, KeyboardInterrupt):
            return
        except Exception as exc:
            if one_shot:
                parser.exit(1, f"Generation failed: {exc}\n")
            print(f"Generation failed: {exc}")


if __name__ == "__main__":
    main()
