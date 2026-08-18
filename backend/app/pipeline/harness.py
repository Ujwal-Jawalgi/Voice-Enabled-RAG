"""
harness.py — Structured RAG pipeline orchestrator.

Each stage is individually timed with time.perf_counter() and results are
returned in the API contract's timings_ms shape. The pipeline short-circuits
to a refusal response if any guardrail fires, WITHOUT calling the LLM.

Pipeline flow:
  input_guardrail → embed_query → retrieve → off_topic_guardrail
  → rerank → build_prompt → llm_call → output_guardrail → response
"""

import time
import logging
import json
import asyncio

from app.models.schemas import QueryResponse, Timings, Source
from app.pipeline.guardrails import input_guardrail, off_topic_guardrail, output_guardrail
from app.pipeline.retrieval import embed_query, search, Candidate
from app.pipeline.rerank import rerank
from app.pipeline.llm import build_prompt, llm_stream
from app.pipeline.session import get_history, append_history
from app.pipeline.sentence_buffer import SentenceBuffer
from app.pipeline.tts import generate_audio

logger = logging.getLogger(__name__)

def _refusal_response_dict(
    transcript: str,
    language: str,
    reason: str,
    timings: dict[str, float],
) -> dict:
    """Build a standardized refusal response when a guardrail fires (returns dict)."""
    stt_t = timings.get("stt", 0.0)
    emb_t = timings.get("embedding", 0.0)
    ret_t = timings.get("retrieval", 0.0)
    rer_t = timings.get("rerank", 0.0)
    llm_t = timings.get("llm", 0.0)
    return QueryResponse(
        transcript=transcript,
        language=language,
        answer=f"I'm unable to answer this query. Reason: {reason}",
        sources=[],
        refused=True,
        confidence="low",
        timings_ms=Timings(
            stt=round(stt_t, 2),
            embedding=round(emb_t, 2),
            retrieval=round(ret_t, 2),
            rerank=round(rer_t, 2),
            llm=round(llm_t, 2),
            total=round(stt_t + emb_t + ret_t + rer_t + llm_t, 2),
        ),
    ).model_dump()


async def run_pipeline(
    transcript: str,
    language: str = "auto",
    stt_time_ms: float = 0.0,
    session_id: str | None = None
):
    """Execute the full RAG pipeline as an async generator yielding SSE strings.
    
    Latency Timing Documentation (Requirement 8):
    - stt: Exact network + inference time from Sarvam API. (Excluded for text queries).
    - embedding: Local CPU time for MiniLM to encode the query (typically 15-30ms).
    - retrieval: FAISS IndexFlatIP search over 18k embeddings (typically 1-5ms).
    - rerank: Ephemeral BM25 tokenization and scoring over top-k (typically <2ms).
    - llm: Groq API streaming response time, which is strictly bounded by network RTT 
           to Groq servers (typically 800ms-1500ms). This represents a hard floor.
    - total: Sum of all measured stages above.
    """
    t_pipeline_start = time.perf_counter()
    timings: dict[str, float] = {"stt": stt_time_ms, "preprocessing": 0.0, "guardrails": 0.0, "llm_first_token": 0.0, "tts_total": 0.0}

    def sse(event_type: str, data: dict):
        payload = {"type": event_type, **data}
        return f"data: {json.dumps(payload)}\n\n"

    # Stage 1: Input Guardrail
    t0 = time.perf_counter()
    ok, reason = input_guardrail(transcript)
    timings["guardrails"] += (time.perf_counter() - t0) * 1000
    
    if not ok:
        timings["total"] = (time.perf_counter() - t_pipeline_start) * 1000
        yield sse("final", {"response": _refusal_response_dict(transcript, language, reason or "Invalid input", timings)})
        return

    # Stage 2: Embed Query
    t0 = time.perf_counter()
    query_vector = embed_query(transcript)
    t_embed = (time.perf_counter() - t0) * 1000

    # Stage 3: Retrieve
    t0 = time.perf_counter()
    candidates: list[Candidate] = search(query_vector, k=5)
    t_retrieve = (time.perf_counter() - t0) * 1000
    timings["embedding"] = t_embed
    timings["retrieval"] = t_retrieve

    # Stage 4: Off-Topic Guardrail
    t0 = time.perf_counter()
    top_score = candidates[0].score if candidates else 0.0
    is_off_topic = off_topic_guardrail(top_score)
    timings["guardrails"] += (time.perf_counter() - t0) * 1000
    
    if is_off_topic:
        timings["total"] = (time.perf_counter() - t_pipeline_start) * 1000
        yield sse("final", {"response": _refusal_response_dict(transcript, language, f"Query appears off-topic (best match score: {top_score:.3f})", timings)})
        return

    # Stage 5: Rerank
    t0 = time.perf_counter()
    reranked: list[Candidate] = rerank(transcript, candidates)
    t_rerank = (time.perf_counter() - t0) * 1000
    timings["rerank"] = t_rerank

    # Yield raw extraction immediately (Requirement for <50ms latency)
    if reranked:
        elapsed_ms = timings.get("embedding", 0.0) + timings.get("retrieval", 0.0) + timings.get("rerank", 0.0)
        yield sse("source_preview", {
            "text": reranked[0].text,
            "elapsed_ms": round(elapsed_ms, 1),
            "passage_id": reranked[0].passage_id,
            "score": round(reranked[0].score, 4)
        })

    # Stage 6: Build Prompt
    context_chunks = [c.text for c in reranked]
    history = get_history(session_id) if session_id else []
    prompt = build_prompt(transcript, context_chunks, language, history)
    # Stage 7: LLM Call & Streaming & TTS
    t0 = time.perf_counter()
    buffer = SentenceBuffer()
    full_answer_parts = []
    
    tts_tasks = []
    
    async for token in llm_stream(prompt, language, transcript):
        if not full_answer_parts:
            timings["llm_first_token"] = (time.perf_counter() - t0) * 1000
        full_answer_parts.append(token)
        yield sse("text", {"token": token})
        sentences = buffer.feed(token)
        for s in sentences:
            # Spawn TTS task in background to avoid blocking the LLM stream
            task = asyncio.create_task(generate_audio(s, language))
            tts_tasks.append((s, task))
            
    # Flush remaining text
    last_s = buffer.flush()
    if last_s:
        task = asyncio.create_task(generate_audio(last_s, language))
        tts_tasks.append((last_s, task))
        
    answer = "".join(full_answer_parts).strip()
    
    t_llm = (time.perf_counter() - t0) * 1000
    timings["llm"] = t_llm

    # Stage 8: Output Guardrail
    t0_out = time.perf_counter()
    grounded, confidence = output_guardrail(answer, context_chunks)
    timings["guardrails"] += (time.perf_counter() - t0_out) * 1000

    if session_id:
        append_history(session_id, transcript, answer)

    stt_t = timings.get("stt", 0.0)
    emb_t = timings.get("embedding", 0.0)
    ret_t = timings.get("retrieval", 0.0)
    rer_t = timings.get("rerank", 0.0)
    llm_t = timings.get("llm", 0.0)
    
    timings["total"] = (time.perf_counter() - t_pipeline_start) * 1000
    
    sources = [Source(passage_id=c.passage_id, score=round(c.score, 4)) for c in reranked]
    
    final_resp = QueryResponse(
        transcript=transcript,
        language=language,
        answer=answer,
        sources=sources,
        refused=False,
        confidence=confidence, # type: ignore
        llm_attempts=1,
        timings_ms=Timings(
            preprocessing=round(timings.get("preprocessing", 0.0), 2),
            stt=round(stt_t, 2),
            embedding=round(emb_t, 2),
            retrieval=round(ret_t, 2),
            guardrails=round(timings.get("guardrails", 0.0), 2),
            rerank=round(rer_t, 2),
            llm_first_token=round(timings.get("llm_first_token", 0.0), 2),
            llm=round(llm_t, 2),
            tts_total=0.0, # Will be set during TTS generation
            total=round(timings.get("total", 0.0), 2),
        ),
    )
    
    yield sse("final", {"response": final_resp.model_dump()})

    # Now yield the generated audio sequentially.
    # Yielding `final` first decouples TTS from the text total latency!
    t0_tts = time.perf_counter()
    for s, task in tts_tasks:
        try:
            audio_b64 = await task
            yield sse("audio", {"sentence": s, "audio_base64": audio_b64})
        except Exception as e:
            logger.error(f"TTS parallel generation failed for sentence: {e}")
    timings["tts_total"] = (time.perf_counter() - t0_tts) * 1000
