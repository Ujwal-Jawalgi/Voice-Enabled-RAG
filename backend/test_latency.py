import urllib.request
import json
import base64
import os

url = 'http://127.0.0.1:8000/query'
headers = {
    'Content-Type': 'application/json'
}

# Create a dummy silent wav file (1 sec) or a short beep.
# Since we need it to recognize speech to not fail with "No speech detected",
# I'll just use a text query first to measure LLM latency, 
# then for STT I need actual audio. Wait, I can just use a text query for LLM!
# But the user asked to optimize STT too. How to get audio?
# I'll download a short english sample or just generate one using gTTS?
# Or I can just check the logs of task-457 where I ran STT tests. Wait, I didn't run STT tests!

# Let's just use text for LLM for now.
def test_text_latency():
    print("Testing LLM latency with text query...")
    text = "What is the capital of Goa?"
    data = json.dumps({'text': text}).encode('utf-8')
    req = urllib.request.Request(url, data=data, headers=headers)
    
    response = urllib.request.urlopen(req)
    body = response.read().decode('utf-8')
    
    for line in body.split("\n"):
        if line.startswith("data: "):
            try:
                parsed = json.loads(line[6:])
                if parsed.get("type") == "final":
                    timings = parsed.get("response").get("timings_ms")
                    print(f"Embedding Latency: {timings.get('embedding')} ms")
                    print(f"Retrieval Latency: {timings.get('retrieval')} ms")
                    print(f"LLM First Token: {timings.get('llm_first_token')} ms")
                    print(f"LLM Total Latency: {timings.get('llm')} ms")
                    print(f"TTS Waiting Time: {timings.get('tts_total')} ms")
                    print(f"Total Text Response Latency: {timings.get('total')} ms")
            except json.JSONDecodeError:
                pass

if __name__ == "__main__":
    print("--- FIRST QUERY ---")
    test_text_latency()
    print("\n--- SECOND QUERY ---")
    test_text_latency()
