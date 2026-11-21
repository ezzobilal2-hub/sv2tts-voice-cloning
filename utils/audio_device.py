from pathlib import Path
import subprocess
import sys


def play_wav(path, blocking=False):
    if sys.platform == "win32":
        import winsound
        flags = winsound.SND_FILENAME | winsound.SND_NODEFAULT
        if not blocking:
            flags |= winsound.SND_ASYNC
        winsound.PlaySound(str(path), flags)
    else:
        import sounddevice as sd
        import soundfile as sf
        wave, rate = sf.read(str(path), dtype="float32")
        sd.play(wave, rate, blocking=blocking)


def record_wav(path, sample_rate, seconds):
    # Some PortAudio drivers terminate the process during initialization.
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), str(path), str(sample_rate), str(seconds)],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, errors="replace",
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "Could not open the default microphone.")


if __name__ == "__main__":
    try:
        import sounddevice as sd
        import soundfile as sf
        path, rate, seconds = sys.argv[1:]
        rate = int(rate)
        wave = sd.rec(int(seconds) * rate, samplerate=rate, channels=1, dtype="float32", blocking=True)
        sf.write(path, wave, rate)
    except Exception as exc:
        sys.exit(str(exc))
