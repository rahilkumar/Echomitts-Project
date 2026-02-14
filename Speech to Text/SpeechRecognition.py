# sudo apt-get install -y python3-pyaudio
# pip3 install vosk

import os
import sys
import json
import time
from array import array
import math
import pyaudio
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
# Audio settings (LOW latency)
# -----------------------------
SAMPLE_RATE = 16000
CHUNK = 512                 # keep low latency like your code
FORMAT = pyaudio.paInt16
CHANNELS = 1

# -----------------------------
# Gate settings (more sensitive = picks up a bit more background)
# -----------------------------
CALIBRATE_SECONDS = 1.0
THRESH_MULT = 1.6           # LOWER = more sensitive (more background)
MIN_THRESHOLD = 90          # LOWER = more sensitive
HANGOVER = 0.18             # slightly higher = keeps listening a bit longer

PRINT_DEBUG = False         # True shows RMS/threshold

def rms_int16(audio_bytes: bytes) -> float:
    """Fast RMS for 16-bit mono PCM using only stdlib."""
    samples = array('h')
    samples.frombytes(audio_bytes)
    n = len(samples)
    if n == 0:
        return 0.0
    acc = 0
    for s in samples:
        acc += s * s
    return math.sqrt(acc / n)

# -----------------------------
# PyAudio init
# -----------------------------
p = pyaudio.PyAudio()
stream = p.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=SAMPLE_RATE,
    input=True,
    frames_per_buffer=CHUNK
)

rec = KaldiRecognizer(model, SAMPLE_RATE)
rec.SetWords(True)

os.system("clear")
print(f"Calibrating ambient noise for {CALIBRATE_SECONDS:.1f}s... stay quiet.")

# -----------------------------
# Calibrate ambient RMS
# -----------------------------
levels = []
t0 = time.time()
while time.time() - t0 < CALIBRATE_SECONDS:
    data = stream.read(CHUNK, exception_on_overflow=False)
    levels.append(rms_int16(data))

levels.sort()
ambient = levels[int(len(levels) * 0.7)] if levels else 0.0
THRESH = max(MIN_THRESHOLD, int(ambient * THRESH_MULT))

print(f"Ambient RMS ≈ {int(ambient)}")
print(f"Threshold  ≈ {THRESH} (mult={THRESH_MULT})")
print("\nSpeak now... (prints FINAL lines only)\n")

last_voice_time = 0.0

try:
    while True:
        data = stream.read(CHUNK, exception_on_overflow=False)

        level = rms_int16(data)
        now = time.time()

        # Gate: only feed recognizer when speech seems active
        if level >= THRESH:
            last_voice_time = now

        speech_active = (now - last_voice_time) <= HANGOVER

        if PRINT_DEBUG:
            sys.stdout.write(f"\rRMS={int(level):4d} TH={THRESH:4d} act={int(speech_active)}   ")
            sys.stdout.flush()

        if not speech_active:
            continue

        # Feed audio; print ONLY finalized text line-by-line
        if rec.AcceptWaveform(data):
            text = json.loads(rec.Result()).get("text", "").strip()
            if text:
                print(text)

except KeyboardInterrupt:
    print("\nStopping...")

finally:
    try:
        stream.stop_stream()
        stream.close()
    except Exception:
        pass
    p.terminate()
