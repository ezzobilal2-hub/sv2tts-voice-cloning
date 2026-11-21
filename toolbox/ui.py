import sys
from pathlib import Path
from time import sleep
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt, QStringListModel
import sounddevice as sd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from encoder.inference import plot_embedding_as_heatmap
from toolbox.utterance import Utterance

default_text = (
    "Welcome to the toolbox! Load an utterance or record one.\n"
    "Once its embedding is created, you can synthesize text.\n"
    "Each line in the text box will be treated as a separate segment.\n"
    "Use the vocoder to generate audio from the synthesized spectrograms."
)

class UI(QDialog):
    min_umap_points = 4
    max_log_lines = 5

    def __init__(self):
        super().__init__()
        # Ensure a single QApplication
        self.app = QApplication.instance() or QApplication(sys.argv)
        self.setWindowTitle("SV2TTS Toolbox")
        self.resize(1200, 750)
        self.logs = []

        # ---------------- UI Widgets ----------------
        self.encoder_box = QComboBox()
        self.synthesizer_box = QComboBox()
        self.vocoder_box = QComboBox()
        self.text_prompt = QPlainTextEdit(default_text)
        self.generate_button = QPushButton("Synthesize & Vocode")
        self.synthesize_button = QPushButton("Synthesize only")
        self.vocode_button = QPushButton("Vocode only")
        self.random_seed_checkbox = QCheckBox("Random seed:")
        self.seed_textbox = QLineEdit("None")

        # Audio playback
        self.waves_cb = QComboBox()
        self.replay_wav_button = QPushButton("Replay")
        self.export_wav_button = QPushButton("Export")

        # Matplotlib plots for spectrograms and embeddings
        self.current_ax = None
        self.gen_ax = None
        self.umap_ax = None

        # ---------------- Layout ----------------
        layout = QVBoxLayout()
        self.setLayout(layout)
        layout.addWidget(self.text_prompt)
        layout.addWidget(self.generate_button)

        # Initialize plots safely
        self._setup_figures(layout)

    def _setup_figures(self, parent_layout):
        """Create matplotlib canvases after QApplication exists"""
        # Current & generated spectrograms
        fig1, axes1 = plt.subplots(1, 2, figsize=(10, 2.25))
        fig1.subplots_adjust(left=0, bottom=0.1, right=1, top=0.8)
        self.current_ax = axes1
        self.current_canvas = FigureCanvas(fig1)
        parent_layout.addWidget(self.current_canvas)

        fig2, axes2 = plt.subplots(1, 2, figsize=(10, 2.25))
        fig2.subplots_adjust(left=0, bottom=0.1, right=1, top=0.8)
        self.gen_ax = axes2
        self.gen_canvas = FigureCanvas(fig2)
        parent_layout.addWidget(self.gen_canvas)

        # UMAP projections
        fig3, ax3 = plt.subplots(figsize=(3, 3))
        fig3.subplots_adjust(left=0.02, bottom=0.02, right=0.98, top=0.98)
        self.umap_ax = ax3
        self.umap_canvas = FigureCanvas(fig3)
        parent_layout.addWidget(self.umap_canvas)

    # ---------------- Audio ----------------
    def play(self, wav, sample_rate):
        try:
            sd.stop()
            sd.play(wav, sample_rate)
        except Exception as e:
            self.log(f"Audio playback error: {e}")

    def stop(self):
        sd.stop()

    # ---------------- Plotting ----------------
    def draw_spec(self, spec, which="current"):
        ax = self.current_ax[1] if which=="current" else self.gen_ax[1]
        ax.clear()
        if spec is not None:
            ax.imshow(spec, aspect="auto", interpolation="none")
            ax.set_title("mel spectrogram")
        if which=="current":
            self.current_canvas.draw()
        else:
            self.gen_canvas.draw()

    def draw_embed(self, embed, name=None, which="current"):
        ax = self.current_ax[0] if which == "current" else self.gen_ax[0]
        ax.clear()
        if embed is not None:
            plot_embedding_as_heatmap(embed, ax=ax, title=name or "embedding")
        if which == "current":
            self.current_canvas.draw()
        else:
            self.gen_canvas.draw()

    # ---------------- Logging ----------------
    def log(self, msg):
        self.logs.append(msg)
        self.logs = self.logs[-self.max_log_lines:]
        print("\n".join(self.logs))

    # ---------------- Populate Models ----------------
    def populate_models(self, models_dir: Path):
        self.encoder_box.clear()
        self.synthesizer_box.clear()
        self.vocoder_box.clear()
        for f in ["encoder.pt", "synthesizer.pt", "vocoder.pt"]:
            path = models_dir / f
            if path.exists():
                if "encoder" in f:
                    self.encoder_box.addItem(str(path))
                elif "synthesizer" in f:
                    self.synthesizer_box.addItem(str(path))
                elif "vocoder" in f:
                    self.vocoder_box.addItem(str(path))
