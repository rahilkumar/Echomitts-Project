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
    print(f"Model '{model_path}' not found.")
    sys.exit(1)

model = Model(model_path)

SAMPLE_RATE = 16000
CHUNK = 320
FORMAT = pyaudio.paInt16
CHANNELS = 1

NOISE_ALPHA = 0.02
SPEECH_MULT = 2.2
MIN_THRESHOLD = 80

SMOOTH_ALPHA = 0.25
HANGOVER = 0.30

PREROLL_SECONDS = 0.25
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
stream = p.open(format=FORMAT, channels=CHANNELS, rate=SAMPLE_RATE,
                input=True, frames_per_buffer=CHUNK)

rec = new_recognizer()

os.system("clear")
print("Live Speech-to-Text (clean output)")
print("Ctrl+C to stop.\n")

noise_est = 150.0
lvl_smooth = noise_est
last_voice_time = 0.0
speech_active = False
last_partial = ""
last_partial_len = 0

def write_live(line: str):
    """Write one-line live caption without creating blank lines."""
    global last_partial_len
    line = line.strip()
    # pad with spaces to overwrite leftovers from previous longer text
    pad = " " * max(0, last_partial_len - len(line))
    sys.stdout.write("\r" + line + pad)
    sys.stdout.flush()
    last_partial_len = len(line)

try:
    while True:
        data = stream.read(CHUNK, exception_on_overflow=False)
        preroll.append(data)

        lvl = rms_int16(data)
        lvl_smooth = (1 - SMOOTH_ALPHA) * lvl_smooth + SMOOTH_ALPHA * lvl

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
            last_partial = ""
            last_partial_len = 0
            # feed pre-roll once
            for chunk in preroll:
                rec.AcceptWaveform(chunk)

        # During speech: feed + show partial live
        if speech_active:
            rec.AcceptWaveform(data)
            partial = json.loads(rec.PartialResult()).get("partial", "").strip()
            if partial and partial != last_partial:
                write_live(partial)
                last_partial = partial

        # Speech ended: print final once, reset recognizer
        if speech_active and not active_now:
            speech_active = False
            final_text = json.loads(rec.FinalResult()).get("text", "").strip()

            # Move to a new line cleanly (no blank gaps)
            sys.stdout.write("\r")
            sys.stdout.flush()
            print(final_text if final_text else "")  # prints one line (can be empty)

            rec = new_recognizer()
            preroll.clear()
            last_partial = ""
            last_partial_len = 0

except KeyboardInterrupt:
    print("\nStopping...")

finally:
    try:
        stream.stop_stream()
        stream.close()
    except Exception:
        pass
    p.terminate()
