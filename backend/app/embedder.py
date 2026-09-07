"""Adapter exposing app.pipeline.retrieval's embedding model under the
eval suite's required interface (embed, embed_one, get_model). Reuses
the already-loaded model/index -- does not load anything twice."""
import numpy as np
from app.pipeline import retrieval as _retrieval


def get_model():
    return _retrieval._model


def embed_one(text: str) -> np.ndarray:
    vec = _retrieval.embed_query(text)  # shape (1, 384)
    return vec[0]


def embed(texts: list[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, 384), dtype=np.float32)
    return _retrieval._model.encode(texts, normalize_embeddings=True).astype(np.float32)
