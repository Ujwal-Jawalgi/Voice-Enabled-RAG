from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import time
import logging
from app.routes import query

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Executing startup warm-up query to page FAISS index into memory...")
    from app.pipeline.retrieval import embed_query, search
    start_time = time.time()
    vec = embed_query("test warm-up query")
    search(vec, k=10)
    elapsed = (time.time() - start_time) * 1000
    logger.info(f"Warm-up query completed in {elapsed:.1f}ms")
    yield

app = FastAPI(title="HH Goa 2026 Voice RAG", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000", 
        "http://127.0.0.1:3000",
        "https://voice-enabled-rag.vercel.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(query.router)

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/debug")
def debug_status():
    from app.pipeline.retrieval import _index, _metadata, _DATA_DIR, _INDEX_PATH, _META_PATH
    import os
    files = []
    try:
        files = os.listdir(_DATA_DIR)
    except Exception as e:
        files = [str(e)]
    return {
        "index_loaded": _index is not None,
        "metadata_len": len(_metadata),
        "data_dir": _DATA_DIR,
        "index_path": _INDEX_PATH,
        "meta_path": _META_PATH,
        "files_in_dir": files
    }

from fastapi import Request
import os
from app.pipeline.retrieval import _DATA_DIR

@app.post("/upload_chunk")
async def upload_chunk(request: Request):
    """Temporary endpoint to bypass Railway SFTP timeout and 100MB HTTP limit."""
    chunk = await request.body()
    with open(os.path.join(_DATA_DIR, "metadata.pkl"), "ab") as f:
        f.write(chunk)
    return {"status": "ok", "bytes_written": len(chunk)}
