from pydantic import BaseModel
from typing import List, Optional, Literal

class Timings(BaseModel):
    preprocessing: float = 0.0
    stt: float = 0.0
    embedding: float = 0.0
    retrieval: float = 0.0
    guardrails: float = 0.0
    rerank: float = 0.0
    llm_first_token: float = 0.0
    llm: float = 0.0
    tts_total: float = 0.0
    total: float = 0.0

class Source(BaseModel):
    passage_id: str
    score: float

class QueryRequest(BaseModel):
    audio_base64: Optional[str] = None
    text: Optional[str] = None
    session_id: Optional[str] = None

class QueryResponse(BaseModel):
    transcript: str
    language: str
    answer: str
    sources: List[Source]
    refused: bool
    confidence: Literal["high", "low"]
    llm_attempts: int = 1
    timings_ms: Timings
