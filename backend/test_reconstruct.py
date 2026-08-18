import faiss
import numpy as np

_INDEX_PATH = r"e:\Voice-RAG\hh-goa-voice-rag\backend\data\vector_index.faiss"

try:
    _faiss_index = faiss.read_index(_INDEX_PATH)
    print("ntotal:", _faiss_index.ntotal)
    _vectors = np.array([_faiss_index.reconstruct(i) for i in range(_faiss_index.ntotal)])
    print("Vectors shape:", _vectors.shape)
except Exception as e:
    print("Error:", e)
