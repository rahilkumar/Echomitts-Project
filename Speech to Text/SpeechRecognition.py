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
CHUNK = 512                 # 512 = very low delay; try 1024 if CPU spikes
FORMAT = pyaudio.paInt16
CHANNELS = 1

# -----------------------------
# Gate settings (tune these)
# -----------------------------
CALIBRATE_SECONDS = 1.0     # quick ambient sample
THRESH_MULT = 2.0           # higher = less sensitive (try 1.6–3.0)
MIN_THRESHOLD = 120         # floor for low RMS mics
HANGOVER = 0.12             # seconds; lower = less delay

PRINT_DEBUG = False         # True shows RMS/threshold to tune

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
print("\nSpeak now...\n")

last_voice_time = 0.0
last_partial = ""

try:
    while True:
        data = stream.read(CHUNK, exception_on_overflow=False)

        level = rms_int16(data)
        now = time.time()

        if level >= THRESH:
            last_voice_time = now

        speech_active = (now - last_voice_time) <= HANGOVER

        if PRINT_DEBUG:
            sys.stdout.write(f"\rRMS={int(level):4d} TH={THRESH:4d} act={int(speech_active)}   ")
            sys.stdout.flush()

        if not speech_active:
            if not PRINT_DEBUG:
                sys.stdout.write("\r(quiet)   ")
                sys.stdout.flush()
            continue

        # Feed audio first, then get partials (fast)
        if rec.AcceptWaveform(data):
            result = json.loads(rec.Result()).get("text", "")
            if result:
                print("\r" + result + " " * 30)
            last_partial = ""
        else:
            partial = json.loads(rec.PartialResult()).get("partial", "")
            if partial and partial != last_partial:
                sys.stdout.write("\r" + partial + " " * 30)
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
