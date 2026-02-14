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

# ---- Audio (low latency) ----
SAMPLE_RATE = 16000
CHUNK = 320                 # ~20ms @ 16kHz (fast). Use 512 if CPU issues.
FORMAT = pyaudio.paInt16
CHANNELS = 1

# ---- Adaptive gate (noise changes loud->quiet) ----
NOISE_ALPHA = 0.02
SPEECH_MULT = 2.2           # higher = less background
MIN_THRESHOLD = 80

# ---- Stability ----
SMOOTH_ALPHA = 0.25
HANGOVER = 0.22             # a bit longer reduces flicker (prevents repeats)

# ---- Pre-roll (don’t miss word beginnings) ----
PREROLL_SECONDS = 0.20
PREROLL_CHUNKS = max(1, int((PREROLL_SECONDS * SAMPLE_RATE) / CHUNK))
preroll = deque(maxlen=PREROLL_CHUNKS)

# ---- Dedup final lines ----
DEDUP_SECONDS = 1.5         # ignore identical final output within this window

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
stream = p.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=SAMPLE_RATE,
    input=True,
    frames_per_buffer=CHUNK
)

rec = KaldiRecognizer(model, SAMPLE_RATE)

os.system("clear")
print("Live Speech-to-Text (no repeats)")
print("Ctrl+C to stop.\n")

noise_est = 150.0
lvl_smooth = noise_est
last_voice_time = 0.0

speech_active = False
preroll_fed_this_segment = False

last_partial = ""

last_final_text = ""
last_final_time = 0.0

try:
    while True:
        data = stream.read(CHUNK, exception_on_overflow=False)
        preroll.append(data)

        lvl = rms_int16(data)
        lvl_smooth = (1 - SMOOTH_ALPHA) * lvl_smooth + SMOOTH_ALPHA * lvl

        # Only learn noise when we are NOT in speech
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
            preroll_fed_this_segment = False
            last_partial = ""

        # Feed pre-roll ONCE per segment
        if speech_active and not preroll_fed_this_segment:
            for chunk in preroll:
                rec.AcceptWaveform(chunk)
            preroll_fed_this_segment = True

        # Speech ended
        if not active_now and speech_active:
            speech_active = False
            preroll_fed_this_segment = False
            last_partial = ""
            continue

        if not speech_active:
            continue

        # Recognize
        if rec.AcceptWaveform(data):
            text = json.loads(rec.Result()).get("text", "").strip()
            if text:
                # Deduplicate identical finals that happen too soon
                if not (text == last_final_text and (now - last_final_time) < DEDUP_SECONDS):
                    # Clear partial line and print final
                    sys.stdout.write("\r" + " " * 120 + "\r")
                    sys.stdout.flush()
                    print(text)
                    last_final_text = text
                    last_final_time = now
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
