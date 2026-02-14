import os
import sys
import json
import time
from array import array
import math
import pyaudio
from vosk import Model, KaldiRecognizer

model_path = "models/vosk-model-small-en-us-0.15/"
if not os.path.exists(model_path):
    print(f"Model '{model_path}' was not found. Please check the path.")
    sys.exit(1)

model = Model(model_path)

SAMPLE_RATE = 16000
CHUNK = 512
FORMAT = pyaudio.paInt16
CHANNELS = 1

CALIBRATE_SECONDS = 1.0
THRESH_MULT = 1.8
MIN_THRESHOLD = 90
HANGOVER = 0.12

def rms_int16(audio_bytes: bytes) -> float:
    samples = array('h')
    samples.frombytes(audio_bytes)
    n = len(samples)
    if n == 0:
        return 0.0
    acc = 0
    for s in samples:
        acc += s * s
    return math.sqrt(acc / n)

p = pyaudio.PyAudio()
stream = p.open(format=FORMAT, channels=CHANNELS, rate=SAMPLE_RATE,
                input=True, frames_per_buffer=CHUNK)

rec = KaldiRecognizer(model, SAMPLE_RATE)

os.system("clear")
print(f"Calibrating ambient noise for {CALIBRATE_SECONDS:.1f}s... stay quiet.")

levels = []
t0 = time.time()
while time.time() - t0 < CALIBRATE_SECONDS:
    data = stream.read(CHUNK, exception_on_overflow=False)
    levels.append(rms_int16(data))

levels.sort()
ambient = levels[int(len(levels) * 0.7)] if levels else 0.0
THRESH = max(MIN_THRESHOLD, int(ambient * THRESH_MULT))

print(f"Ambient RMS ≈ {int(ambient)}")
print(f"Threshold  ≈ {THRESH}")
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

        if (now - last_voice_time) > HANGOVER:
            last_partial = ""
            continue

        # Feed audio
        if rec.AcceptWaveform(data):
            text = json.loads(rec.Result()).get("text", "").strip()
            if text:
                # finalize as a clean line
                print("\r" + " " * 80 + "\r", end="")   # clear partial line
                print(text)
            last_partial = ""
        else:
            # instant feedback (no latency feel)
            partial = json.loads(rec.PartialResult()).get("partial", "")
            if partial and partial != last_partial:
                sys.stdout.write("\r" + partial + " " * 20)
                sys.stdout.flush()
                last_partial = partial

except KeyboardInterrupt:
    print("\nStopping...")

finally:
    stream.stop_stream()
    stream.close()
    p.terminate()
