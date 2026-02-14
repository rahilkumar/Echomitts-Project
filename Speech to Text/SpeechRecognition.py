import os
import sys
import json
import time
from array import array
import math
from collections import deque
import pyaudio
from vosk import Model, KaldiRecognizer

model_path = "models/vosk-model-small-en-us-0.15/"
if not os.path.exists(model_path):
    print(f"Model '{model_path}' was not found. Please check the path.")
    sys.exit(1)

model = Model(model_path)

# Low-latency audio
SAMPLE_RATE = 16000
CHUNK = 512
FORMAT = pyaudio.paInt16
CHANNELS = 1

# Gate tuning (so it doesn't miss words)
CALIBRATE_SECONDS = 1.0
THRESH_MULT = 1.7          # slightly less strict than before
MIN_THRESHOLD = 90
HANGOVER = 0.18            # a bit longer so it doesn't drop mid-word
SMOOTH_ALPHA = 0.25        # 0..1 higher = reacts faster

# Pre-roll: keep last N chunks and feed them when speech starts
PREROLL_SECONDS = 0.25
PREROLL_CHUNKS = max(1, int((PREROLL_SECONDS * SAMPLE_RATE) / CHUNK))

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
print(f"Threshold  ≈ {THRESH} (mult={THRESH_MULT})")
print(f"Pre-roll   ≈ {PREROLL_SECONDS:.2f}s ({PREROLL_CHUNKS} chunks)")
print("\nSpeak now...\n")

last_voice_time = 0.0
last_partial = ""
speech_active = False
level_smooth = float(THRESH)

preroll = deque(maxlen=PREROLL_CHUNKS)

try:
    while True:
        data = stream.read(CHUNK, exception_on_overflow=False)
        preroll.append(data)

        level = rms_int16(data)
        # Smooth the level so the gate doesn't flicker and cut words
        level_smooth = (1 - SMOOTH_ALPHA) * level_smooth + SMOOTH_ALPHA * level

        now = time.time()
        if level_smooth >= THRESH:
            last_voice_time = now

        active_now = (now - last_voice_time) <= HANGOVER

        # If speech just started, feed pre-roll so we don't miss the first syllable
        if active_now and not speech_active:
            for chunk in preroll:
                rec.AcceptWaveform(chunk)
            speech_active = True

        # If speech ended, reset partial tracking
        if not active_now and speech_active:
            speech_active = False
            last_partial = ""
            continue

        if not active_now:
            continue

        # Feed audio & show partials fast; print final line when Vosk finalizes
        if rec.AcceptWaveform(data):
            text = json.loads(rec.Result()).get("text", "").strip()
            if text:
                # clear any partial line and print final
                sys.stdout.write("\r" + " " * 100 + "\r")
                sys.stdout.flush()
                print(text)
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
