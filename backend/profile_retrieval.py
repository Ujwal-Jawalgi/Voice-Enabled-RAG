import time
import os

# Force single threading for math libs to prevent thread thrashing
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

from app.pipeline.retrieval import embed_query, search, _index

def run_profile():
    print(f"OMP_NUM_THREADS = {os.environ.get('OMP_NUM_THREADS')}")
    
    # Warmup
    vec = embed_query("hello world")
    search(vec, k=10)
    
    # Profile embedding
    t0 = time.perf_counter()
    vec = embed_query("What is Voice RAG and how does it optimize latency?")
    t_embed = (time.perf_counter() - t0) * 1000
    print(f"Embedding time: {t_embed:.2f} ms")
    
    # Profile faiss alone
    t0 = time.perf_counter()
    candidates = search(vec, 10)
    t_search = (time.perf_counter() - t0) * 1000
    print(f"FAISS search time: {t_faiss:.2f} ms")
    
    # Profile full search function
    t0 = time.perf_counter()
    cands = search(vec, k=10)
    t_search = (time.perf_counter() - t0) * 1000
    print(f"Full search() time: {t_search:.2f} ms")

if __name__ == "__main__":
    for i in range(3):
        print(f"\nRun {i+1}:")
        run_profile()
