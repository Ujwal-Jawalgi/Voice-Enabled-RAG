from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import query, stt

app = FastAPI(title="HH Goa 2026 Voice RAG")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(query.router)
app.include_router(stt.router)

@app.get("/health")
def health_check():
    return {"status": "ok"}
