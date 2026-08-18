import urllib.request
import json
import time
import numpy as np

url = 'http://127.0.0.1:8000/query'
headers = {'Content-Type': 'application/json'}

queries = [
    "What is Voice RAG?",
    "How does the FAISS index work here?",
    "Explain reciprocal rank fusion.",
    "What is the capital of Goa?",
    "Tell me about latency optimization."
]

results = {
    "embedding": [],
    "retrieval": [],
    "guardrails": [],
    "llm_first_token": [],
    "llm": [],
    "total": []
}

def run_benchmark(num_runs=10):
    print(f"Running benchmark with {num_runs} queries...")
    for i in range(num_runs):
        text = queries[i % len(queries)]
        data = json.dumps({'text': text}).encode('utf-8')
        req = urllib.request.Request(url, data=data, headers=headers)
        
        try:
            start_time = time.perf_counter()
            response = urllib.request.urlopen(req)
            body = response.read().decode('utf-8')
            
            for line in body.split("\n"):
                if line.startswith("data: "):
                    try:
                        parsed = json.loads(line[6:])
                        if parsed.get("type") == "final":
                            timings = parsed.get("response").get("timings_ms")
                            results["embedding"].append(timings.get("embedding", 0))
                            results["retrieval"].append(timings.get("retrieval", 0))
                            results["guardrails"].append(timings.get("guardrails", 0))
                            results["llm_first_token"].append(timings.get("llm_first_token", 0))
                            results["llm"].append(timings.get("llm", 0))
                            results["total"].append(timings.get("total", 0))
                    except json.JSONDecodeError:
                        pass
        except Exception as e:
            print(f"Error on query {i}: {e}")
            
def print_stats(name, values):
    if not values:
        return
    arr = np.array(values)
    print(f"{name:<20} | Min: {np.min(arr):>6.1f} | Avg: {np.mean(arr):>6.1f} | P50: {np.percentile(arr, 50):>6.1f} | P70: {np.percentile(arr, 70):>6.1f} | Max: {np.max(arr):>6.1f}")

if __name__ == "__main__":
    # Warmup query
    run_benchmark(1)
    
    # Reset results
    for k in results.keys():
        results[k] = []
        
    # Run 10 benchmark queries
    run_benchmark(10)
    
    print("\n--- BENCHMARK RESULTS (ms) ---")
    print_stats("Embedding", results["embedding"])
    print_stats("Retrieval", results["retrieval"])
    print_stats("Guardrails", results["guardrails"])
    print_stats("LLM First Token", results["llm_first_token"])
    print_stats("LLM Total", results["llm"])
    print_stats("Total Text Latency", results["total"])
