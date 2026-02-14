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
CHUNK = 320                 # ~20ms @ 16kHz. Use 512 if needed.
FORMAT = pyaudio.paInt16
CHANNELS = 1

# ---- Adaptive gate (handles loud -> quiet changes) ----
NOISE_ALPHA = 0.02
SPEECH_MULT = 2.2           # higher = ignore more background (try 2.0–2.8)
MIN_THRESHOLD = 80

# ---- Stability (avoid flicker) ----
SMOOTH_ALPHA = 0.25
HANGOVER = 0.28             # a bit longer helps avoid on/off while speaking

# ---- Pre-roll (don’t miss beginnings) ----
PREROLL_SECONDS = 0.20
PREROLL_CHUNKS = max(1, int((PREROLL_SECONDS * SAMPLE_RATE) / CHUNK))
preroll = deque(maxlen=PREROLL_CHUNKS)

def rms_int16(audio_bytes: bytes) -> float:
    samples = array('h')
    samples.frombytes(audio_bytes)
    if not samples:
        return 0.0
    acc = 0
    for s in samples:
        acc += s * s
    return math.sqrt(acc / len(samples))

def new_recognizer():
    r = KaldiRecognizer(model, SAMPLE_RATE)
    r.SetWords(True)
    return r

p = pyaudio.PyAudio()
stream = p.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=SAMPLE_RATE,
    input=True,
    frames_per_buffer=CHUNK
)

rec = new_recognizer()

os.system("clear")
print("Live Speech-to-Text (prints ONE line when you stop speaking)")
print("Ctrl+C to stop.\n")

noise_est = 150.0
lvl_smooth = noise_est
last_voice_time = 0.0

speech_active = False
fed_preroll = False
last_partial = ""

try:
    while True:
        data = stream.read(CHUNK, exception_on_overflow=False)
        preroll.append(data)

        lvl = rms_int16(data)
        lvl_smooth = (1 - SMOOTH_ALPHA) * lvl_smooth + SMOOTH_ALPHA * lvl

        # Update noise estimate only when not speaking
        if not speech_active:
            noise_est = (1 - NOISE_ALPHA) * noise_est + NOISE_ALPHA * lvl_smooth

        threshold = max(MIN_THRESHOLD, noise_est * SPEECH_MULT)

        now = time.time()
        if lvl_smooth >= threshold:
            last_voice_time = now

        active_now = (now - last_voice_time) <= HANGOVER

        # Speech started
        if active_now and not speech_active:
            speech_active = True
            fed_preroll = False
            last_partial = ""
            # feed pre-roll ONCE at start so we don't miss first syllable
            for chunk in preroll:
                rec.AcceptWaveform(chunk)
            fed_preroll = True

        # During speech: feed audio + show partial captions (live)
        if speech_active:
            rec.AcceptWaveform(data)  # we feed it, but we DO NOT print Result() here

            partial = json.loads(rec.PartialResult()).get("partial", "")
            if partial and partial != last_partial:
                sys.stdout.write("\r" + partial + " " * 40)
                sys.stdout.flush()
                last_partial = partial

        # Speech ended: print ONE final line and reset recognizer
        if speech_active and (not active_now):
            speech_active = False

            final_text = json.loads(rec.FinalResult()).get("text", "").strip()

            # Clear the caption line
            sys.stdout.write("\r" + " " * 140 + "\r")
            sys.stdout.flush()

            if final_text:
                print(final_text)

            # Reset for next utterance (prevents duplicate finals)
            rec = new_recognizer()
            preroll.clear()
            last_partial = ""
            continue

except KeyboardInterrupt:
    print("\nStopping...")

finally:
    try:
        stream.stop_stream()
        stream.close()
    except Exception:
        pass
    p.terminate()
