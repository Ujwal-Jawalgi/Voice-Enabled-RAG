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

from app.models.schemas import QueryResponse, Timings, Source
from app.pipeline.guardrails import input_guardrail, off_topic_guardrail, output_guardrail
from app.pipeline.retrieval import embed_query, search, Candidate
from app.pipeline.rerank import rerank
from app.pipeline.llm import build_prompt, llm_call

logger = logging.getLogger(__name__)


def _refusal_response(
    transcript: str,
    language: str,
    reason: str,
    timings: dict[str, float],
) -> QueryResponse:
    """Build a standardized refusal response when a guardrail fires."""
    stt_t = timings.get("stt", 0.0)
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
            retrieval=round(ret_t, 2),
            rerank=round(rer_t, 2),
            llm=round(llm_t, 2),
            total=round(stt_t + ret_t + rer_t + llm_t, 2),
        ),
    )


async def run_pipeline(
    transcript: str,
    language: str = "auto",
    stt_time_ms: float = 0.0,
) -> QueryResponse:
    """Execute the full RAG pipeline as discrete typed steps.

    Args:
        transcript: The user's query text (either from STT or direct text input).
        language: Detected language code from STT, or "auto" for text input.
        stt_time_ms: Time already spent on STT (passed through to timings).

    Returns:
        QueryResponse matching the frozen API contract.
    """
    t_pipeline_start = time.perf_counter()
    timings: dict[str, float] = {"stt": stt_time_ms}

    # ------------------------------------------------------------------
    # Stage 1: Input Guardrail
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    ok, reason = input_guardrail(transcript)
    t_input_guard = (time.perf_counter() - t0) * 1000
    logger.info("input_guardrail: ok=%s, reason=%s (%.2fms)", ok, reason, t_input_guard)

    if not ok:
        timings["total"] = (time.perf_counter() - t_pipeline_start) * 1000
        return _refusal_response(transcript, language, reason or "Invalid input", timings)

    # ------------------------------------------------------------------
    # Stage 2: Embed Query
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    query_vector = embed_query(transcript)
    t_embed = (time.perf_counter() - t0) * 1000
    logger.info("embed_query: %.2fms", t_embed)

    # ------------------------------------------------------------------
    # Stage 3: Retrieve
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    candidates: list[Candidate] = search(query_vector, k=10)
    t_retrieve = (time.perf_counter() - t0) * 1000
    timings["retrieval"] = t_embed + t_retrieve  # embedding + FAISS search combined
    logger.info("retrieve: %d candidates in %.2fms (embed=%.2f + search=%.2f)",
                len(candidates), timings["retrieval"], t_embed, t_retrieve)

    # ------------------------------------------------------------------
    # Stage 4: Off-Topic Guardrail (SHORT-CIRCUIT — no LLM call if refused)
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    top_score = candidates[0].score if candidates else 0.0
    is_off_topic = off_topic_guardrail(top_score)
    t_offtopic_guard = (time.perf_counter() - t0) * 1000
    logger.info("off_topic_guardrail: off_topic=%s, top_score=%.4f (%.2fms)",
                is_off_topic, top_score, t_offtopic_guard)

    if is_off_topic:
        timings["total"] = (time.perf_counter() - t_pipeline_start) * 1000
        return _refusal_response(
            transcript, language,
            f"Query appears off-topic (best match score: {top_score:.3f})",
            timings,
        )

    # ------------------------------------------------------------------
    # Stage 5: Rerank
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    reranked: list[Candidate] = rerank(transcript, candidates)
    t_rerank = (time.perf_counter() - t0) * 1000
    timings["rerank"] = t_rerank
    logger.info("rerank: %d -> %d candidates in %.2fms",
                len(candidates), len(reranked), t_rerank)

    # ------------------------------------------------------------------
    # Stage 6: Build Prompt
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    context_chunks = [c.text for c in reranked]
    prompt = build_prompt(transcript, context_chunks, language)
    t_prompt = (time.perf_counter() - t0) * 1000
    logger.info("build_prompt: %.2fms", t_prompt)

    # ------------------------------------------------------------------
    # Stage 7: LLM Call
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    answer, llm_attempts = await llm_call(prompt)
    t_llm = (time.perf_counter() - t0) * 1000
    timings["llm"] = t_llm
    logger.info("llm_call: %.2fms", t_llm)

    # ------------------------------------------------------------------
    # Stage 8: Output Guardrail — grounding check
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    grounded, confidence = output_guardrail(answer, context_chunks)
    t_output_guard = (time.perf_counter() - t0) * 1000
    logger.info("output_guardrail: grounded=%s, confidence=%s (%.2fms)",
                grounded, confidence, t_output_guard)

    # ------------------------------------------------------------------
    # Build Final Response
    # ------------------------------------------------------------------
    stt_t = timings.get("stt", 0.0)
    ret_t = timings.get("retrieval", 0.0)
    rer_t = timings.get("rerank", 0.0)
    llm_t = timings.get("llm", 0.0)

    sources = [
        Source(passage_id=c.passage_id, score=round(c.score, 4))
        for c in reranked
    ]

    return QueryResponse(
        transcript=transcript,
        language=language,
        answer=answer,
        sources=sources,
        refused=False,
        confidence=confidence, # type: ignore
        llm_attempts=llm_attempts,
        timings_ms=Timings(
            stt=round(stt_t, 2),
            retrieval=round(ret_t, 2),
            rerank=round(rer_t, 2),
            llm=round(llm_t, 2),
            total=round(stt_t + ret_t + rer_t + llm_t, 2),
        ),
    )
