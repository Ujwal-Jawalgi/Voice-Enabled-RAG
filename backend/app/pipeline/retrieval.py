"""
retrieval.py — FAISS index + embedding model, loaded ONCE at module import.

Design rationale:
- Module-level singletons avoid per-request model loading (~2s cold start each).
- We normalize query vectors at encode time so IndexFlatIP gives cosine similarity.
- The metadata list is positionally aligned with the FAISS index rows (row i -> metadata[i]).
"""

import os
# Force single threading for math libs to prevent thread thrashing
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import pickle
import logging
from dataclasses import dataclass

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths — resolved relative to the project root, not the app package
# ---------------------------------------------------------------------------
_DEFAULT_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")
_DATA_DIR = os.environ.get("RAILWAY_VOLUME_MOUNT_PATH", _DEFAULT_DATA_DIR)
_INDEX_PATH = os.path.join(_DATA_DIR, "vector_index.faiss")
_META_PATH = os.path.join(_DATA_DIR, "metadata.pkl")
_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class Candidate:
    """A single retrieval result with its metadata."""
    passage_id: str
    language: str
    text: str
    score: float              # cosine similarity (from IndexFlatIP over L2-normed vecs)
    chunk_strategy: str       # "passage" or "fixed_overlap"
    source_query_id: str = ""


# ---------------------------------------------------------------------------
# Module-level singletons — loaded once when the module is first imported.
# On a typical Railway container with 1-2 GB RAM this is fine:
#   - FAISS IndexFlatIP for 18k × 384-dim ≈ 28 MB
#   - Metadata list of 18k dicts ≈ 15-25 MB
#   - MiniLM model ≈ 130 MB
# ---------------------------------------------------------------------------
try:
    logger.info("Loading FAISS index from %s", _INDEX_PATH)
    _faiss_index: faiss.IndexFlatIP = faiss.read_index(_INDEX_PATH) # type: ignore
    
    # Bypass FAISS search on Windows/CPU due to severe OpenMP threading overhead (150-300ms).
    # Extract vectors via reconstruct() to perform pure NumPy dot-product (<10ms).
    _vectors = np.array([_faiss_index.reconstruct(i) for i in range(_faiss_index.ntotal)])
    _index = True
    logger.info("FAISS index loaded and vectors extracted: %d vectors, dimension %d", _faiss_index.ntotal, _faiss_index.d)

    logger.info("Loading metadata from %s", _META_PATH)
    with open(_META_PATH, "rb") as _f:
        _metadata: list[dict] = pickle.load(_f)
    assert len(_metadata) == _faiss_index.ntotal, (
        f"Metadata length ({len(_metadata)}) != index size ({_faiss_index.ntotal})"
    )
except Exception as e:
    logger.warning("Could not load index/metadata. App will start empty. Error: %s", e)
    _index = None
    _vectors = None
    _metadata = []

logger.info("Loading SentenceTransformer model: %s", _MODEL_NAME)
torch.set_grad_enabled(False)
_model = SentenceTransformer(_MODEL_NAME)
_model.eval()

# Force PyTorch graph compilation to avoid 1s latency hit on first query
logger.info("Warming up embedding model...")
_model.encode(["warmup"], normalize_embeddings=True)

logger.info("Retrieval module ready.")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def embed_query(text: str) -> np.ndarray:
    """Embed a single query string into a normalized 384-d vector.

    Returns shape (1, 384).
    Normalization ensures dot product computes cosine similarity.
    """
    vec = _model.encode([text], normalize_embeddings=True)
    return vec.astype(np.float32) # type: ignore


def search(query_vector: np.ndarray, k: int = 5) -> list[Candidate]:
    """Search the exact vector index for the k nearest neighbours."""
    if _index is None or _vectors is None:
        logger.warning("Index is empty or missing, returning no results.")
        return []
        
    # Exact dot product search over 18k vectors is extremely fast in NumPy
    # and bypasses FAISS OpenMP threading issues on Windows.
    scores = np.dot(_vectors, query_vector[0])
    
    # Get top k indices
    indices = np.argpartition(scores, -k)[-k:]
    # Sort the top k by score descending
    indices = indices[np.argsort(-scores[indices])]

    results: list[Candidate] = []
    for rank in range(k):
        idx = int(indices[rank])
        score = float(scores[idx])

        meta = _metadata[idx]
        results.append(Candidate(
            passage_id=meta["passage_id"],
            language=meta["language"],
            text=meta["text"],
            score=score,
            chunk_strategy=meta["chunk_strategy"],
            source_query_id=meta.get("source_query_id", ""),
        ))

    return results


def get_all_texts() -> list[str]:
    """Return all passage texts in index order. Used by rerank.py to build
    the BM25 index at startup without re-loading the pickle."""
    return [m["text"] for m in _metadata]
