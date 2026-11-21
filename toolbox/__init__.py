import sys
import traceback
from pathlib import Path

import torch
import numpy as np

from encoder import inference as encoder
from synthesizer.inference import Synthesizer
from vocoder import inference as vocoder
from toolbox.ui import UI
from toolbox.utterance import Utterance

MAX_WAVS = 15

class Toolbox:
    def __init__(self, datasets_root: Path, models_dir: Path, seed: int = None, cpu: bool = False):
        sys.excepthook = self.excepthook
        self.datasets_root = datasets_root
        self.cpu = cpu
        self.utterances = set()
        self.current_generated = (None, None, None, None)
        self.synthesizer = None
        self.current_wav = None
        self.waves_list = []
        self.waves_count = 0
        self.waves_namelist = []

        # Device selection
        self.device = torch.device("cpu" if cpu else "cuda" if torch.cuda.is_available() else "cpu")
        print(f"Running on device: {self.device}")

        # UI
        self.ui = UI()
        self.reset_ui(models_dir, seed)
        self.setup_events()
        self.ui.start()

    def excepthook(self, exc_type, exc_value, exc_tb):
        traceback.print_exception(exc_type, exc_value, exc_tb)
        self.ui.log("Exception: %s" % exc_value)

    def setup_events(self):
        self.ui.encoder_box.currentIndexChanged.connect(self.init_encoder)
        self.ui.synthesizer_box.currentIndexChanged.connect(lambda: setattr(self, "synthesizer", None))
        self.ui.vocoder_box.currentIndexChanged.connect(self.init_vocoder)
        self.ui.synthesize_button.clicked.connect(self.synthesize)
        self.ui.vocode_button.clicked.connect(self.vocode)
        self.ui.generate_button.clicked.connect(lambda: (self.synthesize(), self.vocode()))
        self.ui.random_seed_checkbox.clicked.connect(self.update_seed_textbox)

    def reset_ui(self, models_dir: Path, seed: int = None):
        self.ui.populate_browser(self.datasets_root)
        self.ui.populate_models(models_dir)
        self.ui.populate_gen_options(seed)

    def init_encoder(self):
        model_fpath = self.ui.current_encoder_fpath
        self.ui.log(f"Loading encoder {model_fpath}...")
        encoder.load_model(model_fpath, device=self.device)
        self.ui.log("Encoder loaded!")

    def init_synthesizer(self):
        model_fpath = self.ui.current_synthesizer_fpath
        self.ui.log(f"Loading synthesizer {model_fpath}...")
        self.synthesizer = Synthesizer(model_fpath)
        self.ui.log("Synthesizer loaded!")

    def init_vocoder(self):
        model_fpath = self.ui.current_vocoder_fpath
        if model_fpath is None:
            self.ui.log("Using Griffin-Lim vocoder")
            return
        self.ui.log(f"Loading vocoder {model_fpath}...")
        vocoder.load_model(model_fpath)
        self.ui.log("Vocoder loaded!")

    def update_seed_textbox(self):
        self.ui.update_seed_textbox()

    def synthesize(self):
        self.ui.log("Generating mel spectrogram...")
        seed = int(self.ui.seed_textbox.text()) if self.ui.random_seed_checkbox.isChecked() else None
        if self.synthesizer is None or seed is not None:
            self.init_synthesizer()

        texts = self.ui.text_prompt.toPlainText().split("\n")
        embed = self.ui.selected_utterance.embed
        embeds = [embed] * len(texts)
        specs = self.synthesizer.synthesize_spectrograms(texts, embeds)
        breaks = [spec.shape[1] for spec in specs]
        spec = np.concatenate(specs, axis=1)

        self.ui.draw_spec(spec, "generated")
        self.current_generated = (self.ui.selected_utterance.speaker_name, spec, breaks, None)

    def vocode(self):
        speaker_name, spec, breaks, _ = self.current_generated
        if spec is None:
            self.ui.log("No spectrogram generated!")
            return
        if not vocoder.is_loaded():
            self.init_vocoder()

        wav = vocoder.infer_waveform(spec)
        wav = wav / np.abs(wav).max() * 0.97
        self.ui.play(wav, Synthesizer.sample_rate)
