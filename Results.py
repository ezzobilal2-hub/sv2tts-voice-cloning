# eval_similarity.py (MOS-based)
import os
import glob
import re
import csv
from pathlib import Path
from tqdm import tqdm

import numpy as np
import soundfile as sf

# Optional metrics
try:
    from pystoi import stoi
except Exception:
    stoi = None

# Encoder and preprocessing from your project
from encoder import inference as encoder
from encoder.audio import preprocess_wav
from utils.default_models import MODELS_DIR, ensure_default_models

SAMPLE_RATE = 16000

PROJECT_DIR = Path(__file__).resolve().parent
REC_DIR = PROJECT_DIR / "recordings"
GEN_DIR = PROJECT_DIR / "generated"
OUT_CSV = "evaluation_results.csv"

# Path to pretrained encoder model
ENCODER_MODEL_PATH = PROJECT_DIR / "saved_models" / "default" / "encoder.pt"


# Utility: extract timestamp from filename
def extract_ts(filename):
    m = re.search(r"_(\d{9,})", filename)
    if m:
        return m.group(1)
    name = Path(filename).stem
    parts = name.split("_")
    if len(parts) >= 2 and parts[-1].isdigit():
        return parts[-1]
    return None

def find_pairs(rec_dir, gen_dir):
    rec_files = sorted(glob.glob(str(rec_dir / "recorded_*.wav")))
    gen_files = sorted(glob.glob(str(gen_dir / "cloned_*.wav")))
    rec_map = {}
    for f in rec_files:
        ts = extract_ts(f)
        if ts:
            rec_map[ts] = f
    pairs = []
    for g in gen_files:
        ts = extract_ts(g)
        if ts and ts in rec_map:
            pairs.append((rec_map[ts], g))
    return pairs

def compute_embedding(wav_path):
    wav, sr = sf.read(wav_path)
    if wav.ndim > 1:
        wav = np.mean(wav, axis=1)
    wav = preprocess_wav(wav, source_sr=sr)
    emb = encoder.embed_utterance(wav)
    return emb

def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

def compute_stoi(ref_wav, gen_wav, sr=SAMPLE_RATE):
    if stoi is None:
        return None
    try:
        return float(stoi(ref_wav, gen_wav, sr, extended=False))
    except Exception:
        return None

def load_and_prep(wav_path):
    wav, sr = sf.read(wav_path)
    if wav.ndim > 1:
        wav = np.mean(wav, axis=1)
    if sr != SAMPLE_RATE:
        try:
            import resampy
            wav = resampy.resample(wav, sr, SAMPLE_RATE)
            sr = SAMPLE_RATE
        except Exception:
            pass
    return wav, sr

def mos_estimate(sim, sscore=None):
    """
    Approximate MOS (1–5):
    - Cosine similarity mapped from 0–1 → 1–5
    - If STOI is available, average both.
    """
    if sim is None:
        return None
    mos_sim = 1 + 4 * max(0, min(1, sim))
    if sscore is not None:
        mos_stoi = 1 + 4 * max(0, min(1, sscore))
        return (mos_sim + mos_stoi) / 2
    return mos_sim

def main():
    pairs = find_pairs(REC_DIR, GEN_DIR)
    if not pairs:
        print("No matched recording/cloned pairs found. Make sure filenames include matching timestamps.")
        return

    # Load encoder model if not already loaded
    if not encoder.is_loaded():
        if ENCODER_MODEL_PATH == MODELS_DIR / "default" / "encoder.pt":
            ensure_default_models(MODELS_DIR, ["encoder"])
        print(f"Loading encoder model from {ENCODER_MODEL_PATH} ...")
        encoder.load_model(ENCODER_MODEL_PATH)

    rows = []
    sims, stois, moss = [], [], []

    for rec_path, gen_path in tqdm(pairs, desc="Evaluating pairs"):
        rec_path, gen_path = str(rec_path), str(gen_path)
        try:
            emb_ref = compute_embedding(rec_path)
            emb_gen = compute_embedding(gen_path)
            sim = cosine_sim(emb_ref, emb_gen)
        except Exception as e:
            print(f"Embedding error for pair {rec_path}, {gen_path}: {e}")
            sim = None

        try:
            ref_wav, _ = load_and_prep(rec_path)
            gen_wav, _ = load_and_prep(gen_path)
            min_len = min(len(ref_wav), len(gen_wav))
            ref_wav = ref_wav[:min_len]
            gen_wav = gen_wav[:min_len]
        except Exception as e:
            print(f"Wave loading error for pair {rec_path}, {gen_path}: {e}")
            ref_wav = gen_wav = None

        sscore = compute_stoi(ref_wav, gen_wav) if ref_wav is not None else None
        mos = mos_estimate(sim, sscore)

        rows.append({
            "recording": rec_path,
            "generated": gen_path,
            "cosine_similarity": sim,
            "stoi": sscore,
            "mos_estimate": mos
        })
        if sim is not None: sims.append(sim)
        if sscore is not None: stois.append(sscore)
        if mos is not None: moss.append(mos)

    with open(OUT_CSV, "w", newline="", encoding="utf8") as f:
        writer = csv.DictWriter(f, fieldnames=["recording", "generated", "cosine_similarity", "stoi", "mos_estimate"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"Saved results to {OUT_CSV}")
    if sims:
        print(f"Cosine similarity: mean={np.mean(sims):.4f} std={np.std(sims):.4f}")
    if stois:
        print(f"STOI: mean={np.mean(stois):.4f} std={np.std(stois):.4f}")
    if moss:
        print(f"MOS estimate: mean={np.mean(moss):.4f} std={np.std(moss):.4f}")


# Pytest-compatible entrypoint
def test_mos_evaluation(tmp_path, monkeypatch):
    if not find_pairs(REC_DIR, GEN_DIR):
        import pytest
        pytest.skip("Add local recording/cloned pairs to run this integration test.")
    monkeypatch.chdir(tmp_path)
    main()
    with Path(OUT_CSV).open(newline="", encoding="utf8") as result:
        rows = list(csv.DictReader(result))
    assert rows and all(row["cosine_similarity"] for row in rows)


if __name__ == "__main__":
    main()
