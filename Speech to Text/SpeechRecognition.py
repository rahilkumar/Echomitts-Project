import os
import sys
import json
import time
from array import array
import math
from collections import deque
import pyaudio
from vosk import Model, KaldiRecognizer

# ---- Model ----
model_path = "models/vosk-model-small-en-us-0.15/"
if not os.path.exists(model_path):
    print(f"Model '{model_path}' not found.")
    sys.exit(1)

model = Model(model_path)

# ---- Low-latency audio ----
SAMPLE_RATE = 16000
CHUNK = 320  # ~20ms @ 16kHz (lower latency than 512). Try 320 or 480.
FORMAT = pyaudio.paInt16
CHANNELS = 1

# ---- Adaptive gate (handles loud->quiet transitions) ----
NOISE_ALPHA = 0.02      # noise adapts speed (0.01–0.05)
SPEECH_MULT = 2.0       # higher = ignore more background (1.6–3.0)
MIN_THRESHOLD = 80

# ---- Smooth + hangover to avoid chopping words ----
SMOOTH_ALPHA = 0.25
HANGOVER = 0.20         # seconds to keep "speech active" after level drops

# ---- Pre-roll so we don't miss the start of words ----
PREROLL_SECONDS = 0.25
PREROLL_CHUNKS = max(1, int((PREROLL_SECONDS * SAMPLE_RATE) / CHUNK))

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
stream = p.open(format=FORMAT, channels=CHANNELS, rate=SAMPLE_RATE,
                input=True, frames_per_buffer=CHUNK)

rec = KaldiRecognizer(model, SAMPLE_RATE)

os.system("clear")
print("Live Speech-to-Text (partial captions + final lines)")
print("Ctrl+C to stop.\n")

noise_est = 150.0
lvl_smooth = noise_est
last_voice_time = 0.0
speech_active = False
last_partial = ""

preroll = deque(maxlen=PREROLL_CHUNKS)

try:
    while True:
        data = stream.read(CHUNK, exception_on_overflow=False)
        preroll.append(data)

        lvl = rms_int16(data)
        lvl_smooth = (1 - SMOOTH_ALPHA) * lvl_smooth + SMOOTH_ALPHA * lvl

        # Update noise estimate only when we believe it's NOT speech
        if not speech_active:
            noise_est = (1 - NOISE_ALPHA) * noise_est + NOISE_ALPHA * lvl_smooth

        threshold = max(MIN_THRESHOLD, noise_est * SPEECH_MULT)

        now = time.time()
        if lvl_smooth >= threshold:
            last_voice_time = now

        active_now = (now - last_voice_time) <= HANGOVER

        # speech just started -> feed pre-roll so we don't miss word beginnings
        if active_now and not speech_active:
            for chunk in preroll:
                rec.AcceptWaveform(chunk)
            speech_active = True

        # speech ended
        if not active_now and speech_active:
            speech_active = False
            last_partial = ""
            continue

        if not active_now:
            continue

        # Feed audio. Show partial captions live.
        if rec.AcceptWaveform(data):
            text = json.loads(rec.Result()).get("text", "").strip()
            if text:
                # Clear caption line and print final as a new line
                sys.stdout.write("\r" + " " * 120 + "\r")
                sys.stdout.flush()
                print(text)
            last_partial = ""
        else:
            partial = json.loads(rec.PartialResult()).get("partial", "")
            if partial and partial != last_partial:
                # Live captions on one line
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
