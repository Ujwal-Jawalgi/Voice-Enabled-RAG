import asyncio
import codecs
from app.pipeline.llm import build_prompt, llm_stream, llm_call

async def main():
    with codecs.open("investigate_hindi_context_out.md", "w", "utf-8") as f:
        f.write("# Hindi Context Investigation\n\n")
        
        query_text = "गोवा की राजधानी क्या है?"
        language = "hindi"
        
        # Simulating FAISS unrelated English context
        unrelated_context = [
            "Voice RAG is an architecture for real-time multilingual querying.",
            "The system uses FAISS for rapid in-memory vector similarity search.",
            "FastAPI serves the backend and communicates via Server-Sent Events (SSE).",
            "This document does not contain any information about geography.",
            "Latencies are heavily optimized to be under 3 seconds end-to-end."
        ]
        
        prompt = build_prompt(query_text, unrelated_context, language)
        f.write("## 1. Unrelated English Context (Refusal Expected)\n")
        
        tokens = []
        async for chunk in llm_stream(prompt):
            tokens.append(chunk)
            
        full_answer = "".join(tokens)
        f.write(f"- Streamed Answer: `{repr(full_answer)}`\n")

if __name__ == "__main__":
    asyncio.run(main())
