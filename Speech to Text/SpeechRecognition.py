# sudo apt-get install -y python3-pyaudio
# sudo pip3 install vosk

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
sample_rate = 16000
chunk_size = 4096            # smaller chunks react faster than 8192
format = pyaudio.paInt16
channels = 1

# -----------------------------
# Background-noise reduction (simple voice gate)
# -----------------------------
RMS_THRESHOLD = 600          # raise to ignore more background (try 400–1500)
SPEECH_HANGOVER = 0.5        # seconds to keep listening after volume drops
last_voice_time = 0.0

def rms_int16(audio_bytes: bytes) -> float:
    """Compute RMS loudness for 16-bit mono audio bytes."""
    count = len(audio_bytes) // 2
    if count <= 0:
        return 0.0
    samples = struct.unpack("<" + "h" * count, audio_bytes)
    acc = 0
    for s in samples:
        acc += s * s
    return math.sqrt(acc / count)

# -----------------------------
# PyAudio init
# -----------------------------
p = pyaudio.PyAudio()

stream = p.open(
    format=format,
    channels=channels,
    rate=sample_rate,
    input=True,
    frames_per_buffer=chunk_size
)

recognizer = KaldiRecognizer(model, sample_rate)

os.system('clear')
print("\nSpeak now... (background gate ON)")
print(f"RMS_THRESHOLD={RMS_THRESHOLD}, HANGOVER={SPEECH_HANGOVER}s\n")

try:
    while True:
        # Read audio (avoid crashing on overflow)
        data = stream.read(chunk_size, exception_on_overflow=False)

        # Measure loudness
        level = rms_int16(data)
        now = time.time()

        # If it's loud enough, treat as "voice present"
        if level >= RMS_THRESHOLD:
            last_voice_time = now

        # If we haven't heard "voice" recently, skip feeding Vosk
        if (now - last_voice_time) > SPEECH_HANGOVER:
            sys.stdout.write("\r(quiet) ")
            sys.stdout.flush()
            continue

        # Feed Vosk only during detected speech
        if recognizer.AcceptWaveform(data):
            result_json = json.loads(recognizer.Result())
            text = result_json.get('text', '')
            if text:
                print("\r" + text + " " * 10)  # extra spaces to overwrite old text
        else:
            partial_json = json.loads(recognizer.PartialResult())
            partial = partial_json.get('partial', '')
            sys.stdout.write('\r' + partial)
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
