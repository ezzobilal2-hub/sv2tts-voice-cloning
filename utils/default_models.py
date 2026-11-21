import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.request import urlopen

from tqdm import tqdm

MODELS_DIR = Path(__file__).resolve().parents[1] / "saved_models"
MODEL_PAGE = "https://huggingface.co/CorentinJ/SV2TTS"
REVISION = "c1b838170d8e75f5c223935de2766f693a19d860"
BASE_URL = f"{MODEL_PAGE}/resolve/{REVISION}"

default_models = {
    "encoder": (f"{BASE_URL}/encoder.pt", 17090379),
    "synthesizer": (f"{BASE_URL}/synthesizer.pt", 370554559),
    "vocoder": (f"{BASE_URL}/vocoder.pt", 53845290),
}
CHECKSUMS = {
    "encoder": "39373b86598fa3da9fcddee6142382efe09777e8d37dc9c0561f41f0070f134e",
    "synthesizer": "c05e07428f95d0ed8755e1ef54cc8ae251300413d94ce5867a56afe39c499d94",
    "vocoder": "1d7a6861589e927e0fbdaa5849ca022258fe2b58a20cc7bfb8fb598ccf936169",
}


def download(url: str, target: Path, bar_pos=0):
    target.parent.mkdir(exist_ok=True, parents=True)
    digest = hashlib.sha256()
    temp_path = None
    try:
        with urlopen(url, timeout=60) as response, NamedTemporaryFile(dir=target.parent, delete=False) as temp:
            temp_path = Path(temp.name)
            with tqdm(total=default_models[target.stem][1], unit="B", unit_scale=True,
                      desc=target.name, position=bar_pos, leave=False) as progress:
                while chunk := response.read(1024 * 1024):
                    temp.write(chunk)
                    digest.update(chunk)
                    progress.update(len(chunk))
        if digest.hexdigest() != CHECKSUMS[target.stem]:
            raise ValueError("SHA-256 checksum mismatch")
        temp_path.replace(target)
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Could not download {target.name}. Place the checkpoint from "
                           f"{MODEL_PAGE}/tree/{REVISION} at {target}.") from exc
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def ensure_default_models(models_dir: Path, names=None):
    with ThreadPoolExecutor(max_workers=len(default_models)) as pool:
        jobs = []
        for name, (url, size) in default_models.items():
            if names is not None and name not in names:
                continue
            target = models_dir / "default" / f"{name}.pt"
            if target.is_file() and target.stat().st_size == size:
                continue
            jobs.append(pool.submit(download, url, target, len(jobs)))
        for job in jobs:
            job.result()
