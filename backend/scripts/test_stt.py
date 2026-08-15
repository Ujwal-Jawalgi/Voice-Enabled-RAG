"""
test_stt.py — Smoke test for the STT-enabled /query endpoint.

Reads a local test_audio.wav file, converts it to base64, and POSTs it
to the FastAPI app using TestClient to simulate a frontend request.
"""

import sys
import os
import base64

# Ensure the backend app package is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.stdout.reconfigure(encoding='utf-8') # type: ignore

from fastapi.testclient import TestClient
from app.main import app

def main():
    audio_path = os.path.join(os.path.dirname(__file__), "test_audio.wav.ogg")
    
    if not os.path.exists(audio_path):
        print(f"ERROR: Audio file not found at {audio_path}")
        print("Please place a test WAV file there and re-run.")
        return

    print(f"Reading audio file from {audio_path}...")
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()

    print("Encoding to base64...")
    audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
    
    # We prepend the data URI prefix to ensure the stt.py stripping logic works properly
    # as the frontend canvas/MediaRecorder often sends it this way.
    payload = {
        "audio_base64": f"data:audio/wav;base64,{audio_base64}"
    }

    print("Sending POST request to /query endpoint...")
    client = TestClient(app)
    
    # TestClient blocks until the async request is fulfilled in its own event loop
    response = client.post("/query", json=payload)
    
    print("\n" + "="*50)
    print("RESPONSE")
    print("="*50)
    
    if response.status_code != 200:
        print(f"HTTP Error {response.status_code}: {response.text}")
        return
        
    data = response.json()
    
    print(f"Transcript : {data.get('transcript')!r}")
    print(f"Language   : {data.get('language')}")
    print(f"Refused    : {data.get('refused')}")
    print(f"Confidence : {data.get('confidence')}")
    print(f"Answer     : {data.get('answer')}")
    
    timings = data.get("timings_ms", {})
    print(f"\nTimings (ms):")
    print(f"  STT      : {timings.get('stt')}")
    print(f"  Retrieval: {timings.get('retrieval')}")
    print(f"  Rerank   : {timings.get('rerank')}")
    print(f"  LLM      : {timings.get('llm')}")
    print(f"  Total    : {timings.get('total')}")
    
if __name__ == "__main__":
    main()
