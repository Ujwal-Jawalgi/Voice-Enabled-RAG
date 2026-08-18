from fastapi import APIRouter, HTTPException
from app.models.schemas import QueryRequest, QueryResponse, Timings
from app.routes.stt import process_audio
from app.pipeline.harness import run_pipeline

from fastapi.responses import StreamingResponse
import json

router = APIRouter(tags=["query"])

# Map Sarvam ISO codes to our internal metadata/LLM instruction tags
LANGUAGE_MAP = {
    "en-IN": "english",
    "hi-IN": "hindi",
    "kn-IN": "kannada"
}

import re

def detect_language(text: str) -> str:
    """
    Detect language based on Unicode block of the text.
    Groups shared scripts (e.g., Devanagari -> hindi, Bengali -> bengali).
    """
    if re.search(r'[\u0900-\u097F]', text):
        return "hindi"      # Covers Hindi, Marathi, Sanskrit, Nepali, Konkani, etc.
    elif re.search(r'[\u0C80-\u0CFF]', text):
        return "kannada"
    elif re.search(r'[\u0980-\u09FF]', text):
        return "bengali"    # Covers Bengali and Assamese
    elif re.search(r'[\u0A80-\u0AFF]', text):
        return "gujarati"
    elif re.search(r'[\u0B80-\u0BFF]', text):
        return "tamil"
    elif re.search(r'[\u0C00-\u0C7F]', text):
        return "telugu"
    elif re.search(r'[\u0D00-\u0D7F]', text):
        return "malayalam"
    elif re.search(r'[\u0B00-\u0B7F]', text):
        return "odia"
    elif re.search(r'[\u0A00-\u0A7F]', text):
        return "punjabi"
    elif re.search(r'[\u0600-\u06FF]', text):
        return "urdu"       # Covers Urdu (and other Arabic-based scripts like Kashmiri)
    else:
        return "english"

@router.post("/query")
async def handle_query(req: QueryRequest):
    if req.audio_base64:
        try:
            transcript, raw_lang, stt_time_ms = await process_audio(req.audio_base64)
            # Map the language or fallback to english
            language = LANGUAGE_MAP.get(raw_lang, "english")
        except ValueError as e:
            # STT failure should just ask the user to retry the recording, returning a typed refusal
            error_msg = str(e)
            async def _error_gen():
                yield f"data: {json.dumps({'type': 'final', 'response': QueryResponse(transcript='', language='auto', answer=error_msg, sources=[], refused=True, confidence='low', timings_ms=Timings(stt=0.0, embedding=0.0, retrieval=0.0, rerank=0.0, llm=0.0, total=0.0)).model_dump()})}\n\n"
            return StreamingResponse(_error_gen(), media_type="text/event-stream")
    elif req.text:
        transcript = req.text
        language = detect_language(transcript)
        stt_time_ms = 0.0
    else:
        raise HTTPException(status_code=400, detail="Must provide either audio_base64 or text")

    # Pass through to the structured harness pipeline which now yields SSE events
    return StreamingResponse(
        run_pipeline(
            transcript=transcript,
            language=language,
            stt_time_ms=stt_time_ms,
            session_id=req.session_id
        ),
        media_type="text/event-stream"
    )
