import asyncio
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from app.pipeline.harness import run_pipeline

async def test_timing():
    print("Loading pipeline... (cold start)")
    
    # Run once to warm up
    await run_pipeline("warm up query")
    print("Warmup complete. Running live query to get steady-state timings...\n")
    
    t0 = time.perf_counter()
    resp = await run_pipeline("what is a corporation?", language="english")
    t1 = time.perf_counter()
    
    print("-" * 50)
    print("STAGE-BY-STAGE TIMING BREAKDOWN")
    print("-" * 50)
    print(f"Embedding (MiniLM CPU): {resp.timings_ms.embedding:.2f} ms")
    print(f"Retrieval (FAISS search): {resp.timings_ms.retrieval:.2f} ms")
    print(f"Rerank (BM25 RRF): {resp.timings_ms.rerank:.2f} ms")
    print("-" * 50)
    
    ret_rer = resp.timings_ms.embedding + resp.timings_ms.retrieval + resp.timings_ms.rerank
    print(f"Total Local Path (Embed+Ret+Rer): {ret_rer:.2f} ms")
    
    print("-" * 50)
    print(f"LLM Generation: {resp.timings_ms.llm:.2f} ms")
    print(f"Total End-to-End: {resp.timings_ms.total:.2f} ms")
    print("-" * 50)

if __name__ == "__main__":
    asyncio.run(test_timing())
