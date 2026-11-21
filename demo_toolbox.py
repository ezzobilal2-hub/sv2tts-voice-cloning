import sys
import threading
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QApplication, QCheckBox, QFileDialog, QLabel, QPlainTextEdit,
    QPushButton, QVBoxLayout, QWidget,
)

from encoder import inference as encoder
from synthesizer.inference import Synthesizer
from utils.audio_device import play_wav, record_wav
from utils.default_models import MODELS_DIR, ensure_default_models
from vocoder import inference as vocoder


PROJECT_DIR = MODELS_DIR.parent
SAMPLE_RATE = Synthesizer.sample_rate


class VoiceCloneApp(QWidget):
    completed = pyqtSignal(str)
    reference_ready = pyqtSignal(object, object)

    def __init__(self, synthesizer):
        super().__init__()
        self.synthesizer = synthesizer
        self.embedding = None
        self.reference = None
        self.setWindowTitle("Real-time Voice Cloner")
        self.text_input = QPlainTextEdit()
        self.text_input.setPlaceholderText("Type a sentence to synthesize")
        self.record_button = QPushButton("Record Voice")
        self.open_button = QPushButton("Open reference WAV")
        self.synthesize_button = QPushButton("Synthesize")
        self.fast = QCheckBox("Fast vocoder")
        self.playback = QCheckBox("Play generated audio")
        self.playback.setChecked(True)
        self.status = QLabel("Record five seconds or open a reference WAV.")
        layout = QVBoxLayout(self)
        for widget in (self.text_input, self.record_button, self.open_button, self.fast,
                       self.playback, self.synthesize_button, self.status):
            layout.addWidget(widget)
        self.record_button.clicked.connect(self.start_recording)
        self.open_button.clicked.connect(self.open_reference)
        self.synthesize_button.clicked.connect(self.start_synthesis)
        self.reference_ready.connect(self.set_reference)
        self.completed.connect(self.finish)
        self.synthesize_button.setEnabled(False)

    def start_worker(self, status, work, *args):
        self.status.setText(status)
        for button in (self.record_button, self.open_button, self.synthesize_button):
            button.setEnabled(False)

        def run():
            try:
                message = work(*args)
            except Exception as exc:
                message = str(exc)
            self.completed.emit(message)

        threading.Thread(target=run, daemon=True).start()

    def finish(self, message):
        self.status.setText(message)
        self.record_button.setEnabled(True)
        self.open_button.setEnabled(True)
        self.synthesize_button.setEnabled(self.embedding is not None)

    def set_reference(self, path, embedding):
        self.reference, self.embedding = path, embedding

    def open_reference(self):
        name, _ = QFileDialog.getOpenFileName(self, "Open reference", "", "WAV files (*.wav)")
        if name:
            self.start_worker("Reading reference…", self.prepare_reference, Path(name))

    def prepare_reference(self, path):
        self.reference_ready.emit(path, encoder.embed_reference(path))
        return f"Reference ready: {path.name}"

    def start_recording(self):
        self.start_worker("Recording for five seconds…", self.record_voice)

    def record_voice(self):
        directory = PROJECT_DIR / "recordings"
        directory.mkdir(exist_ok=True)
        path = directory / f"recorded_{time.time_ns()}.wav"
        record_wav(path, SAMPLE_RATE, 5)
        return self.prepare_reference(path)

    def start_synthesis(self):
        lines = [s.strip() for s in self.text_input.toPlainText().splitlines() if s.strip()]
        if not lines:
            self.status.setText("Enter a sentence to synthesize.")
            return
        self.start_worker("Synthesizing…", self.synthesize_text, lines, self.reference,
                          self.embedding, self.fast.isChecked(), self.playback.isChecked())

    def synthesize_text(self, lines, reference, embedding, fast, playback):
        specs = self.synthesizer.synthesize_spectrograms(lines, [embedding] * len(lines))
        wav = vocoder.infer_waveform(np.concatenate(specs, axis=1), fast=fast)
        peak = np.max(np.abs(wav))
        if peak > 0:
            wav = wav * (0.97 / peak)
        directory = PROJECT_DIR / "generated"
        directory.mkdir(exist_ok=True)
        basename = reference.stem.removeprefix("recorded_")
        path = directory / f"cloned_{basename}.wav"
        if path.exists():
            path = directory / f"cloned_{basename}_{time.time_ns()}.wav"
        sf.write(str(path), wav, SAMPLE_RATE)
        if playback:
            try:
                play_wav(path)
            except Exception as exc:
                return f"Saved {path.name}. Playback failed: {exc}"
        return f"Saved {path.name}"


def main():
    app = QApplication(sys.argv)
    if not torch.cuda.is_available():
        torch.set_num_threads(min(torch.get_num_threads(), 3))
    ensure_default_models(MODELS_DIR)
    directory = MODELS_DIR / "default"
    encoder.load_model(directory / "encoder.pt")
    synthesizer = Synthesizer(directory / "synthesizer.pt")
    vocoder.load_model(directory / "vocoder.pt")
    window = VoiceCloneApp(synthesizer)
    window.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
