import os
import faiss
import pickle
import sys
from sentence_transformers import SentenceTransformer

def main():
    sys.stdout.reconfigure(encoding='utf-8') # type: ignore
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    index_path = os.path.join(data_dir, "vector_index.faiss")
    meta_path = os.path.join(data_dir, "metadata.pkl")

    print(f"Loading FAISS index from {index_path}...")
    index = faiss.read_index(index_path)
    
    print(f"Loading metadata from {meta_path}...")
    with open(meta_path, "rb") as f:
        metadata = pickle.load(f)

    print("Loading SentenceTransformer model (paraphrase-multilingual-MiniLM-L12-v2)...")
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    import json
    with open("query_412499.json", "r", encoding="utf-8") as f:
        q_412499 = json.load(f)["q"]
        
    queries = [
        "What is the capital of France?",
        "फ्रांस की राजधानी क्या है?", # Hindi: What is the capital of France?
        "ಫ್ರಾನ್ಸ್ ರಾಜಧಾನಿ ಯಾವುದು?", # Kannada: What is the capital of France?
        q_412499
    ]
    for q in queries:
        print(f"\n{'='*60}\nQuery: {q}\n{'='*60}")
        # Encode and normalize the query for Inner Product similarity
        emb = model.encode([q], normalize_embeddings=True)
        
        # Search FAISS index
        k = 5
        distances, indices = index.search(emb, k)
        
        for i in range(k):
            idx = indices[0][i]
            score = distances[0][i]
            
            if idx != -1 and idx < len(metadata):
                meta = metadata[idx]
                text = meta["text"]
                # Truncate text for cleaner terminal output
                if len(text) > 250:
                    text = text[:247] + "..."
                lang = meta.get("language", "unknown")
                print(f"[Rank {i+1}] Score: {score:.4f} | Lang: {lang} | ID: {meta.get('passage_id')}")
                print(f"Text: {text}\n")

if __name__ == "__main__":
    main()
