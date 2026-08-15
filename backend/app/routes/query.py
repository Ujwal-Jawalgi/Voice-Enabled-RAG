from fastapi import APIRouter, HTTPException
from app.models.schemas import QueryRequest, QueryResponse, Timings
from app.routes.stt import process_audio
from app.pipeline.harness import run_pipeline

router = APIRouter(tags=["query"])

# Map Sarvam ISO codes to our internal metadata/LLM instruction tags
LANGUAGE_MAP = {
    "en-IN": "english",
    "hi-IN": "hindi",
    "kn-IN": "kannada"
}

@router.post("/query", response_model=QueryResponse)
async def handle_query(req: QueryRequest):
    if req.audio_base64:
        try:
            transcript, raw_lang, stt_time_ms = await process_audio(req.audio_base64)
            # Map the language or fallback to english
            language = LANGUAGE_MAP.get(raw_lang, "english")
        except ValueError as e:
            # STT failure should just ask the user to retry the recording, returning a typed refusal
            return QueryResponse(
                transcript="",
                language="auto",
                answer=str(e),
                sources=[],
                refused=True,
                confidence="low",
                timings_ms=Timings(stt=0.0, retrieval=0.0, rerank=0.0, llm=0.0, total=0.0)
            )
    elif req.text:
        transcript = req.text
        language = "auto"
        stt_time_ms = 0.0
    else:
        raise HTTPException(status_code=400, detail="Must provide either audio_base64 or text")

    # Pass through to the structured harness pipeline
    return await run_pipeline(
        transcript=transcript,
        language=language,
        stt_time_ms=stt_time_ms
    )
