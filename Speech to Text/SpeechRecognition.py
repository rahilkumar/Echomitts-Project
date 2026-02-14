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
CHUNK = 320  # ~20ms @ 16kHz (low latency). Try 480 if CPU is high.
FORMAT = pyaudio.paInt16
CHANNELS = 1

# ---- Adaptive gate ----
NOISE_ALPHA = 0.02
SPEECH_MULT = 2.0
MIN_THRESHOLD = 80

SMOOTH_ALPHA = 0.25
HANGOVER = 0.22

# ---- Pre-roll ----
PREROLL_SECONDS = 0.25
PREROLL_CHUNKS = max(1, int((PREROLL_SECONDS * SAMPLE_RATE) / CHUNK))

# ---- Anti-repeat settings ----
DEDUP_WINDOW_SEC = 1.5      # if same/similar final appears within this window, ignore
MIN_FINAL_CHARS = 3         # ignore super-short finals like "uh" / blanks
SIMILARITY_THRESHOLD = 0.90 # 0..1, higher = stricter "same sentence"

def rms_int16(audio_bytes: bytes) -> float:
    samples = array('h')
    samples.frombytes(audio_bytes)
    if not samples:
        return 0.0
    acc = 0
    for s in samples:
        acc += s * s
    return math.sqrt(acc / len(samples))

def normalize_text(s: str) -> str:
    # Basic normalization to catch repeats with punctuation/case differences
    s = s.strip().lower()
    s = " ".join(s.split())
    return s

def similarity(a: str, b: str) -> float:
    """
    Lightweight similarity: Jaccard on word sets.
    Works well enough to detect repeated sentences without extra libraries.
    """
    aw = set(a.split())
    bw = set(b.split())
    if not aw or not bw:
        return 0.0
    return len(aw & bw) / len(aw | bw)

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
print("Live Speech-to-Text (with anti-repeat)")
print("Ctrl+C to stop.\n")

noise_est = 150.0
lvl_smooth = noise_est
last_voice_time = 0.0
speech_active = False
last_partial = ""

# track last printed final to avoid duplicates
last_final_norm = ""
last_final_time = 0.0

preroll = deque(maxlen=PREROLL_CHUNKS)

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

        # speech just started -> feed pre-roll so we don't miss beginnings
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

        # Feed recognizer
        if rec.AcceptWaveform(data):
            text = json.loads(rec.Result()).get("text", "").strip()
            if not text:
                continue

            norm = normalize_text(text)
            if len(norm) < MIN_FINAL_CHARS:
                continue

            # Deduplicate repeated finals
            too_soon = (now - last_final_time) < DEDUP_WINDOW_SEC
            is_similar = similarity(norm, last_final_norm) >= SIMILARITY_THRESHOLD if last_final_norm else False

            if too_soon and is_similar:
                # ignore duplicate
                continue

            # Clear caption line and print final
            sys.stdout.write("\r" + " " * 120 + "\r")
            sys.stdout.flush()
            print(text)

            last_final_norm = norm
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
