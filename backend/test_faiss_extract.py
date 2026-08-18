import faiss
import os

_INDEX_PATH = r"e:\Voice-RAG\hh-goa-voice-rag\backend\data\vector_index.faiss"

try:
    _faiss_index = faiss.read_index(_INDEX_PATH)
    print("Type of index:", type(_faiss_index))
    
    print("Trying reconstruct_n...")
    try:
        _vectors = _faiss_index.reconstruct_n(0, _faiss_index.ntotal)
        print("reconstruct_n succeeded! shape:", _vectors.shape)
    except Exception as e:
        print("reconstruct_n failed:", e)

    print("Trying downcast...")
    try:
        _downcast = faiss.downcast_index(_faiss_index)
        _vectors = faiss.rev_swig_ptr(_downcast.get_xb(), _downcast.ntotal * _downcast.d).reshape(_downcast.ntotal, _downcast.d)
        print("downcast + rev_swig_ptr succeeded! shape:", _vectors.shape)
    except Exception as e:
        print("downcast failed:", e)
        
except Exception as e:
    print("General exception:", e)
