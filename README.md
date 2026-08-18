# HH Goa 2026 Voice RAG

# 🎙️ Pikachu  Voice-Enabled Multilingual RAG System

#RAGInGoa

**Built for HH Goa 2026 - Shortlisting Task 2**

Pikachu is a voice-enabled, multilingual Retrieval-Augmented Generation (RAG) system. Ask a question out loud (or type it) in any of **15 Indic languages**, and it transcribes your speech, retrieves the most relevant passages from a **7,00,000-chunk unified vector index**, and generates a grounded, guardrailed answer - with full latency transparency at every stage.

🌐 **Live Demo:** `https://voice-enabled-rag.vercel.app/`
📂 **Backend Repo:** `pikchau-rag.up.railway.app`

---

## 📖 Overview

Most RAG demos hide the real cost of using hosted APIs. Pikachu doesn't. Every millisecond shown on the dashboard — STT, embedding, retrieval, rerank, LLM generation — is a real, measured number from `time.perf_counter( ), with no hardcoded.

The result is a system that's honest about where the time goes: a **fully optimized local pipeline** (embedding + retrieval + rerank) running in just hundred milliseconds across a 700k-vector index, alongside two **mandatory hosted API calls** (speech-to-text and LLM generation) whose network-bound latency is reported plainly rather than concealed.

---

## ✨ Features

### 🎤 Voice Input
* Record a question via the browser mic (MediaRecorder API)
* Speech-to-text via **Sarvam AI's `saarika` model**, purpose-built for Indic languages and code-mixing
* Live on-screen transcript with detected language shown immediately

### 🌏 Multilingual Retrieval
* **One unified FAISS index** (not 15 separate ones) spanning **15 languages**: English, Hindi, Kannada, Assamese, Bengali, Gujarati, Malayalam, Marathi, Nepali, Odia, Punjabi, Sanskrit, Tamil, Telugu, and Urdu
* ~7,00,000 indexed chunks total, built from the `ai4bharat/MSMARCO-XI` dataset
* A single shared multilingual embedding space (`paraphrase-multilingual-MiniLM-L12-v2`) enables **native cross-lingual retrieval** - a query in one language can surface a passage tagged in another

### 🧩 Multi-Strategy Chunking
* **Passage-level chunking** (primary) - MS MARCO passages are already short, natural semantic units.
* **Fixed-size chunking with overlap** (256 tokens, 40-token overlap) - applied automatically as a fallback for any unusually long passage.
* Every chunk is tagged in metadata with which strategy produced it.

### 🛡️ Guardrails
* **Input guardrail** - rejects empty, too-short, or unsafe input (including prompt-injection patterns) before any embedding or retrieval happens
* **Off-topic guardrail** - refuses to call the LLM at all when the top FAISS similarity score falls below an empirically tuned threshold, instead of risking a hallucinated answer
* **Output guardrail** - a lexical-overlap grounding check on the generated answer against retrieved context, flagging low-confidence answers
* **Empty-response safety net** - catches rare model-level generation failures (observed specifically in cross-lingual refusal paths) and substitutes a localized fallback message rather than returning nothing

### ⚡ Two-Stage Answer Display
* The top retrieved source is shown **immediately** after retrieval completes, clearly labeled as retrieved context - not as the answer
* The AI-generated answer streams in below it once generation completes, clearly labeled separately

### 📊 Transparent Latency Reporting
* Every pipeline stage - STT, embedding, retrieval, rerank, LLM - is individually timed and displayed
* P50/P70/P100 latency and Recall@5 benchmarked across a real query sample and documented in [`docs/LATENCY_REPORT.md`](docs/LATENCY_REPORT.md)
* Known bottlenecks and their root causes are documented honestly, including fixes already applied (see below)

---

## 🏗️ Architecture

```
Browser mic --audio--> /stt (Sarvam saarika) --transcript + language-->
    input_guardrail (unsafe/empty check)
    --> embed_query (MiniLM, local, in-process)
    --> FAISS search (k=10, unified 700k-vector index)
    --> off_topic_guardrail (refuse + skip LLM if below threshold)
    --> BM25 + dense RRF rerank (top 3–5 candidates)
    --> prompt builder (language-aware, grounded-answer template)
    --> Groq LLM call (timeout + 1 retry)
    --> output_guardrail (lexical grounding check)
    --> structured JSON response (answer, sources, confidence, timings_ms)
    --> Frontend renders: source preview → generated answer → latency breakdown
```

No RAG orchestration framework (LangChain/LlamaIndex) is used - the retrieval, rerank, and harness logic is hand-written in plain Python for full control over the latency-critical path and to keep the implementation defensible and explainable.

---

## 🛠️ Tech Stack

| Layer | Choice |
|---|---|
| Frontend | Next.js (App Router) + TypeScript, Tailwind CSS + shadcn/ui |
| Backend | FastAPI (Python 3.11) |
| Speech-to-Text | Sarvam AI (`saarika` model) |
| LLM | Groq API — `openai/gpt-oss-20b` |
| Embeddings | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (local, loaded once at startup) |
| Vector Index | FAISS `IndexFlatIP`, in-process, in-memory |
| Reranker | BM25 (`rank_bm25`) + dense cosine, fused via Reciprocal Rank Fusion |
| Dataset | `ai4bharat/MSMARCO-XI` (validation split) via Hugging Face `datasets` |
| Testing | pytest + custom `benchmark.py` (P50/P70/P100 + Recall@5) |
| Deployment | Frontend → Vercel · Backend → Railway (Docker) |

> **Note on the LLM:** the project originally targeted `llama-3.1-8b-instant`. Groq deprecated this model mid-project (announced June 17, 2026); the pipeline was migrated to Groq's officially recommended replacement, `openai/gpt-oss-20b`, with the prompt and `max_tokens` retuned accordingly.

---

## 📈 Latency — The Honest Version

We report the full end-to-end picture, not a cherry-picked sub-metric.

| Stage | Typical time |
|---|---|
| Embedding (local, CPU) | ~5–9 ms |
| Retrieval (FAISS, 690k vectors, exact search) | ~10–15 ms |
| Rerank (BM25 + RRF) | ~5–10 ms |
| Speech-to-Text (Sarvam, network) | ~100-110 ms |
| LLM generation (Groq, network) | ~15–20 ms |
| **Total end-to-end (voice query)** | **~100–150 ms** |

What *is* fully under our control - embedding, retrieval, and reranking across a 7,00,000-vector, 15-language index - is optimized and sits in the low hundreds of milliseconds.


Full methodology and per-query benchmark data: [`docs/LATENCY_REPORT.md`](docs/LATENCY_REPORT.md)

---

## 📂 Repository Structure

```
pikachu/
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── routes/          # stt.py, query.py
│   │   ├── pipeline/        # harness.py, guardrails.py, retrieval.py, rerank.py, llm.py
│   │   ├── models/          # pydantic schemas
│   │   └── config.py
│   ├── scripts/
│   │   ├── build_index.py   # offline chunk + embed + FAISS build
│   │   └── benchmark.py     # P50/P70/P100 + Recall@5
│   ├── data/                 # FAISS index + metadata (gitignored)
│   ├── tests/
│   └── requirements.txt
├── frontend/                 # Next.js app
├── docs/
│   ├── ARCHITECTURE.md
│   ├── CHUNKING_STRATEGY.md
│   └── LATENCY_REPORT.md
└── README.md
```

---

## 🧪 Testing

* **pytest** covers guardrail edge cases (empty/unsafe input, off-topic threshold behavior), rerank fusion ordering, and an end-to-end `/query` integration test
* **`benchmark.py`** replays a real query sample from the dataset, reports P50/P70/P100 per stage, and computes Recall@5 against ground-truth passage pairs

```bash
cd backend
pytest -v
python scripts/benchmark.py
```

---

## 🚀 Running Locally

```bash
# Backend
cd backend
pip install -r requirements.txt
python scripts/build_index.py     # builds the FAISS index (first time only)
uvicorn app.main:app --reload

# Frontend
cd frontend
npm install
npm run dev
```

Environment variables required: `SARVAM_API_KEY`, `GROQ_API_KEY`.

---

## 👥 Team members- Ujwal & Srilakshmi 

Built by a two-person team over a 4-day strech:
* **Backend / RAG / AI:** dataset pipeline, chunking, embeddings, FAISS index, retrieval, rerank, harness, guardrails, Sarvam + Groq integration, benchmarking, backend deployment
* **Frontend / Integration:** Next.js UI, API client layer, frontend deployment, documentation, demo production

---
