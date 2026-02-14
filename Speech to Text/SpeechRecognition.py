# sudo apt-get install -y python3-pyaudio
# sudo pip3 install vosk numpy

import os
import sys
import json
import time
import math
import struct
import pyaudio
import numpy as np
from vosk import Model, KaldiRecognizer

# -----------------------------
# Vosk model path
# -----------------------------
model_path = "models/vosk-model-small-en-us-0.15/"
if not os.path.exists(model_path):
    print(f"Model '{model_path}' was not found. Please check the path.")
    sys.exit(1)

model = Model(model_path)

# -----------------------------
# Audio settings
# -----------------------------
sample_rate = 16000
chunk_size = 1024          # lower = less delay; 1024 is fast on Pi 5
format = pyaudio.paInt16
channels = 1

# -----------------------------
# Adaptive gate settings
# -----------------------------
CALIBRATE_SECONDS = 2.0    # stay quiet
THRESH_MULT = 2.5          # higher = less sensitive (try 2.0–3.5)
MIN_RMS = 120              # floor threshold for low RMS mics
SPEECH_HANGOVER = 0.25     # shorter = less delay
BAND_RATIO_TH = 0.35       # higher = stricter speech-only (try 0.30–0.55)

# Optional: see tuning values live
PRINT_DEBUG = True

def rms_int16(audio_bytes: bytes) -> float:
    n = len(audio_bytes) // 2
    if n <= 0:
        return 0.0
    samples = struct.unpack("<" + "h" * n, audio_bytes)
    acc = 0
    for s in samples:
        acc += s * s
    return math.sqrt(acc / n)

def speech_band_ratio(audio_bytes: bytes, sr: int) -> float:
    """
    Returns fraction of energy in ~300–3000 Hz band vs total energy.
    Helps reject low-frequency rumble and some non-speech noise.
    """
    x = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
    if x.size == 0:
        return 0.0

    # window to reduce spectral leakage
    x *= np.hanning(x.size)

    spec = np.fft.rfft(x)
    mag2 = (spec.real**2 + spec.imag**2)

    freqs = np.fft.rfftfreq(x.size, d=1.0/sr)

    total = mag2.sum()
    if total <= 0:
        return 0.0

    band = mag2[(freqs >= 300) & (freqs <= 3000)].sum()
    return float(band / total)

# -----------------------------
# PyAudio init
# -----------------------------
p = pyaudio.PyAudio()
stream = p.open(
    format=format,
    channels=channels,
    rate=sample_rate,
    input=True,
    frames_per_buffer=chunk_size
)

recognizer = KaldiRecognizer(model, sample_rate)
recognizer.SetWords(True)

os.system("clear")

# -----------------------------
# Calibrate ambient RMS
# -----------------------------
print(f"Calibrating ambient noise for {CALIBRATE_SECONDS:.1f}s... stay quiet.")
levels = []
start = time.time()
while time.time() - start < CALIBRATE_SECONDS:
    data = stream.read(chunk_size, exception_on_overflow=False)
    levels.append(rms_int16(data))

levels.sort()
ambient = levels[int(len(levels) * 0.7)] if levels else 0.0
RMS_THRESHOLD = max(MIN_RMS, int(ambient * THRESH_MULT))

print(f"Ambient RMS ≈ {int(ambient)}")
print(f"RMS_THRESHOLD ≈ {RMS_THRESHOLD}  (mult={THRESH_MULT})")
print(f"BAND_RATIO_TH = {BAND_RATIO_TH}")
print("\nSpeak now... (FAST partial + better background rejection)\n")

last_voice_time = 0.0
last_partial = ""

try:
    while True:
        data = stream.read(chunk_size, exception_on_overflow=False)
        level = rms_int16(data)
        ratio = speech_band_ratio(data, sample_rate)
        now = time.time()

        # Decide if this chunk looks like "speech nearby"
        looks_like_speech = (level >= RMS_THRESHOLD) and (ratio >= BAND_RATIO_TH)

        if looks_like_speech:
            last_voice_time = now

        speech_active = (now - last_voice_time) <= SPEECH_HANGOVER

        if PRINT_DEBUG:
            sys.stdout.write(f"\rRMS={int(level)} TH={RMS_THRESHOLD}  ratio={ratio:.2f}  act={int(speech_active)}   ")
            sys.stdout.flush()

        if not speech_active:
            # don’t feed Vosk when we think it's just background
            if not PRINT_DEBUG:
                sys.stdout.write("\r(quiet)   ")
                sys.stdout.flush()
            continue

        # Feed Vosk (FAST partial output)
        if recognizer.AcceptWaveform(data):
            result = json.loads(recognizer.Result()).get("text", "")
            if result:
                print("\r" + result + " " * 20)
            last_partial = ""
        else:
            partial = json.loads(recognizer.PartialResult()).get("partial", "")
            if partial and partial != last_partial:
                sys.stdout.write("\r" + partial + " " * 20)
                sys.stdout.flush()
                last_partial = partial

except KeyboardInterrupt:
    print("\nStopping...")

finally:
    try:
        stream.stop_stream()
        stream.close()
    except Exception:
        pass
    p.terminate()
