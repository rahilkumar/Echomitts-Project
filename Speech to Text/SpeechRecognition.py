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
    print("Model not found.")
    sys.exit(1)

model = Model(model_path)

SAMPLE_RATE = 16000
CHUNK = 512
FORMAT = pyaudio.paInt16
CHANNELS = 1

# Adaptive tuning
NOISE_ALPHA = 0.02        # how fast noise estimate adapts (lower = slower)
SPEECH_MULT = 2.0         # how much louder than noise to trigger
MIN_THRESHOLD = 80
HANGOVER = 0.18
SMOOTH_ALPHA = 0.25

PREROLL_SECONDS = 0.3
PREROLL_CHUNKS = int((PREROLL_SECONDS * SAMPLE_RATE) / CHUNK)

def rms_int16(audio_bytes: bytes) -> float:
    samples = array('h')
    samples.frombytes(audio_bytes)
    if not samples:
        return 0.0
    acc = 0
    for s in samples:
        acc += s * s
    return math.sqrt(acc / len(samples))

p = pyaudio.PyAudio()
stream = p.open(format=FORMAT, channels=CHANNELS,
                rate=SAMPLE_RATE, input=True,
                frames_per_buffer=CHUNK)

rec = KaldiRecognizer(model, SAMPLE_RATE)

os.system("clear")
print("Adaptive speech recognition running...\n")

noise_estimate = 200.0   # starting guess
level_smooth = noise_estimate
last_voice_time = 0.0
speech_active = False
last_partial = ""

preroll = deque(maxlen=PREROLL_CHUNKS)

try:
    while True:
        data = stream.read(CHUNK, exception_on_overflow=False)
        preroll.append(data)

        level = rms_int16(data)

        # Smooth level
        level_smooth = (1 - SMOOTH_ALPHA) * level_smooth + SMOOTH_ALPHA * level

        # Update background noise estimate ONLY when not speaking
        if not speech_active:
            noise_estimate = (1 - NOISE_ALPHA) * noise_estimate + NOISE_ALPHA * level_smooth

        threshold = max(MIN_THRESHOLD, noise_estimate * SPEECH_MULT)

        now = time.time()

        if level_smooth >= threshold:
            last_voice_time = now

        active_now = (now - last_voice_time) <= HANGOVER

        # Speech started
        if active_now and not speech_active:
            for chunk in preroll:
                rec.AcceptWaveform(chunk)
            speech_active = True

        # Speech ended
        if not active_now and speech_active:
            speech_active = False
            last_partial = ""
            continue

        if not active_now:
            continue

        # Recognition
        if rec.AcceptWaveform(data):
            text = json.loads(rec.Result()).get("text", "").strip()
            if text:
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
    stream.stop_stream()
    stream.close()
    p.terminate()
