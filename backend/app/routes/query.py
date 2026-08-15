from fastapi import APIRouter
from app.models.schemas import QueryRequest, QueryResponse, Timings

router = APIRouter(tags=["query"])

@router.post("/query", response_model=QueryResponse)
async def handle_query(req: QueryRequest):
    # This will hook into app.pipeline.harness
    return QueryResponse(
        transcript="mock transcript",
        language="en",
        answer="mock answer",
        sources=[],
        refused=False,
        confidence="high",
        timings_ms=Timings(stt=0, retrieval=0, rerank=0, llm=0, total=0)
    )
