# sudo apt-get install -y python3-pyaudio
# pip3 install vosk

import os
import sys
import json
import time
import audioop
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
CHUNK = 1024                 # 1024 is low delay; try 512 if your CPU can handle it
FORMAT = pyaudio.paInt16
CHANNELS = 1

# -----------------------------
# Gate settings (tune these)
# -----------------------------
CALIBRATE_SECONDS = 1.0      # quick ambient sample
THRESH_MULT = 2.2            # higher = less sensitive (try 2.0–3.5)
MIN_THRESHOLD = 120          # floor for low RMS mics
HANGOVER = 0.20              # seconds; lower = stops listening quicker

PRINT_DEBUG = False          # True shows RMS/threshold to tune

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
# Calibrate ambient RMS (FAST)
# -----------------------------
levels = []
t0 = time.time()
while time.time() - t0 < CALIBRATE_SECONDS:
    data = stream.read(CHUNK, exception_on_overflow=False)
    levels.append(audioop.rms(data, 2))   # width=2 bytes for int16

levels.sort()
ambient = levels[int(len(levels) * 0.7)] if levels else 0
THRESH = max(MIN_THRESHOLD, int(ambient * THRESH_MULT))

print(f"Ambient RMS ≈ {ambient}")
print(f"Threshold  ≈ {THRESH} (mult={THRESH_MULT})")
print("\nSpeak now...\n")

last_voice_time = 0.0
last_partial = ""

try:
    while True:
        data = stream.read(CHUNK, exception_on_overflow=False)

        rms = audioop.rms(data, 2)
        now = time.time()

        if rms >= THRESH:
            last_voice_time = now

        speech_active = (now - last_voice_time) <= HANGOVER

        if PRINT_DEBUG:
            sys.stdout.write(f"\rRMS={rms:4d} TH={THRESH:4d} act={int(speech_active)}   ")
            sys.stdout.flush()

        if not speech_active:
            if not PRINT_DEBUG:
                sys.stdout.write("\r(quiet)   ")
                sys.stdout.flush()
            continue

        # IMPORTANT: feed audio first, then ask for partial
        if rec.AcceptWaveform(data):
            result = json.loads(rec.Result()).get("text", "")
            if result:
                print("\r" + result + " " * 20)
            last_partial = ""
        else:
            partial = json.loads(rec.PartialResult()).get("partial", "")
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
