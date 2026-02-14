# Install (one-time):
# sudo apt-get update
# sudo apt-get install -y python3-pyaudio
# pip3 install vosk

import os
import sys
import json
import time
import math
import struct
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
# Audio settings
# -----------------------------
SAMPLE_RATE = 16000
CHUNK_SIZE = 4096
FORMAT = pyaudio.paInt16
CHANNELS = 1

# -----------------------------
# Voice gate settings (tuned for arm-length USB mic)
# -----------------------------
CALIBRATE_SECONDS = 2.0      # stay quiet while this runs
THRESH_MULTIPLIER = 4.5      # higher = less sensitive (try 4.0–6.0)
MIN_THRESHOLD = 250          # floor so it doesn't get too sensitive
SPEECH_HANGOVER = 0.45       # seconds to keep listening after speech drops
PRINT_LEVELS = True         # True = show RMS + threshold live (for tuning)

def rms_int16(audio_bytes: bytes) -> float:
    """RMS loudness for 16-bit mono audio."""
    n = len(audio_bytes) // 2
    if n <= 0:
        return 0.0
    samples = struct.unpack("<" + "h" * n, audio_bytes)
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
    frames_per_buffer=CHUNK_SIZE
)

recognizer = KaldiRecognizer(model, SAMPLE_RATE)
recognizer.SetWords(True)

os.system("clear")

# -----------------------------
# Calibrate ambient noise
# -----------------------------
print(f"Calibrating ambient noise for {CALIBRATE_SECONDS:.1f}s... stay quiet.")
levels = []
start = time.time()

while time.time() - start < CALIBRATE_SECONDS:
    data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
    levels.append(rms_int16(data))

levels.sort()
# robust ambient estimate: 70th percentile avoids spikes
ambient = levels[int(len(levels) * 0.7)] if levels else 0.0
threshold = max(MIN_THRESHOLD, int(ambient * THRESH_MULTIPLIER))

print(f"Ambient RMS ≈ {int(ambient)}")
print(f"Gate threshold ≈ {threshold}  (multiplier={THRESH_MULTIPLIER})")
print("\nSpeak now...\n")

last_voice_time = 0.0

try:
    while True:
        data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
        level = rms_int16(data)
        now = time.time()

        if PRINT_LEVELS:
            sys.stdout.write(f"\rRMS={int(level)}  TH={threshold}     ")
            sys.stdout.flush()

        # Detect voice by loudness
        if level >= threshold:
            last_voice_time = now

        # If no recent voice, skip feeding Vosk (reduces background pickup)
        if (now - last_voice_time) > SPEECH_HANGOVER:
            if not PRINT_LEVELS:
                sys.stdout.write("\r(quiet)     ")
                sys.stdout.flush()
            continue

        # Feed Vosk only during detected speech
        if recognizer.AcceptWaveform(data):
            result = json.loads(recognizer.Result())
            text = result.get("text", "")
            if text:
                print("\r" + text + " " * 10)
        else:
            if not PRINT_LEVELS:
                partial = json.loads(recognizer.PartialResult()).get("partial", "")
                sys.stdout.write("\r" + partial)
                sys.stdout.flush()

except KeyboardInterrupt:
    print("\nStopping...")

finally:
    try:
        stream.stop_stream()
        stream.close()
    except Exception:
        pass
    p.terminate()
