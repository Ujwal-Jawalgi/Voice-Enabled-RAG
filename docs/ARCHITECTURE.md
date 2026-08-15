# Voice-RAG Architecture

## Repo Structure
- `backend/`: Python FastAPI backend.
  - `app/`: Main application code.
    - `main.py`: FastAPI application entry point and routes registration.
    - `routes/`: API endpoint handlers (e.g., `/query` and `/stt`).
    - `models/`: Pydantic schemas defining the API contract.
    - `pipeline/`: Core RAG logic.
      - `harness.py`: Orchestrates the end-to-end pipeline.
      - `retrieval.py`: FAISS dense retrieval and metadata management.
      - `rerank.py`: Reciprocal Rank Fusion (RRF) and ephemeral BM25 scoring.
      - `llm.py`: Groq API integration and prompt building.
      - `guardrails.py`: Input, off-topic, and output grounding guardrails.
  - `scripts/`: Offline processing scripts.
    - `build_index.py`: Parses MSMARCO-XI datasets, chunks texts, generates embeddings, and saves FAISS and metadata artifacts.
    - `benchmark.py`: Runs rigorous latency and recall benchmarking.
  - `data/`: Local storage for `vector_index.faiss`, `metadata.pkl`, and log files.
  - `tests/`: Pytest suite for pipeline logic and API endpoints.

## Tech Stack
- **API Framework**: FastAPI
- **Audio Processing / STT**: Sarvam API
- **Embedding Model**: `paraphrase-multilingual-MiniLM-L12-v2` (SentenceTransformers)
- **Vector Database**: FAISS (`IndexFlatIP` for inner product / cosine similarity)
- **Lexical Retriever**: BM25 (`rank_bm25`)
- **LLM**: Llama 3.1 8B via Groq API
- **Testing**: Pytest

## Pipeline Stages
1. **Speech-to-Text (Optional)**: Audio is sent to the Sarvam API for transcription and language detection.
2. **Input Guardrail**: The query is validated for length and disallowed patterns.
3. **Embedding**: The text query is converted to a dense vector.
4. **Retrieval**: FAISS retrieves the top 10 candidates using cosine similarity.
5. **Off-Topic Guardrail**: If the top dense score is below `0.35`, the query is refused.
6. **Rerank**: An ephemeral BM25 index is built over the 10 candidates. Dense and lexical ranks are fused using Reciprocal Rank Fusion (RRF) to select the top 5 passages.
7. **Prompt Building & LLM**: The top 5 passages are injected into a prompt and sent to Groq. If the call times out (4.0s), it is retried once.
8. **Output Guardrail**: A lexical overlap check ensures the generated answer is grounded in the retrieved context.

## API Contract
**Endpoint**: `POST /query`

**Request (`QueryRequest`)**:
- `audio_base64` (string, optional): Base64 encoded audio.
- `text` (string, optional): Text query (must provide either audio or text).

**Response (`QueryResponse`)**:
- `transcript` (string): The recognized or provided text query.
- `language` (string): Detected language.
- `answer` (string): The generated response or refusal reason.
- `sources` (list): Up to 5 `Source` objects containing `passage_id`, `text`, and `score`.
- `refused` (boolean): True if guardrails blocked the query.
- `confidence` (string): `"high"` or `"low"`.
- `llm_attempts` (integer): Number of LLM API attempts (1 or 2).
- `timings_ms` (`Timings` object): Breakdown of latency (`stt`, `retrieval`, `rerank`, `llm`, `total`).
