from pydantic import BaseModel
from typing import List, Optional, Literal

class Timings(BaseModel):
    stt: float
    retrieval: float
    rerank: float
    llm: float
    total: float

class Source(BaseModel):
    passage_id: str
    score: float

class QueryRequest(BaseModel):
    audio_base64: Optional[str] = None
    text: Optional[str] = None

class QueryResponse(BaseModel):
    transcript: str
    language: str
    answer: str
    sources: List[Source]
    refused: bool
    confidence: Literal["high", "low"]
    llm_attempts: int = 1
    timings_ms: Timings
