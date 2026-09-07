import os
import pickle
import faiss
import numpy as np
from collections import Counter
import time

def main():
    t0 = time.time()
    
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    old_index_path = os.path.join(data_dir, "vector_index.faiss")
    old_meta_path = os.path.join(data_dir, "metadata.pkl")
    
    new_index_path = os.path.join(data_dir, "vector_index_smaller.faiss")
    new_meta_path = os.path.join(data_dir, "metadata_smaller.pkl")
    
    # 1. Load existing
    print(f"Loading existing FAISS index from {old_index_path}...")
    old_index = faiss.read_index(old_index_path)
    
    print(f"Loading existing metadata from {old_meta_path}...")
    with open(old_meta_path, "rb") as f:
        old_meta = pickle.load(f)
        
    assert len(old_meta) == old_index.ntotal, "Metadata length does not match index size!"
    
    # 2. Extract and filter
    caps = {
        "english": 50000,
        "hindi": 50000,
        "kannada": 50000,
    }
    default_cap = 7000
    
    counts = Counter()
    strategy_counts = Counter()
    
    new_meta = []
    new_vectors = []
    
    print("Filtering rows...")
    for i in range(old_index.ntotal):
        meta = old_meta[i]
        lang = meta["language"]
        cap = caps.get(lang, default_cap)
        
        if counts[lang] < cap:
            new_meta.append(meta)
            new_vectors.append(old_index.reconstruct(i))
            counts[lang] += 1
            strategy_counts[meta["chunk_strategy"]] += 1
            
    # 3. Build new index
    new_vectors_np = np.array(new_vectors, dtype=np.float32)
    dimension = new_vectors_np.shape[1]
    
    print("Building new FAISS index...")
    new_index = faiss.IndexFlatIP(dimension)
    new_index.add(new_vectors_np)
    
    # 4. Save
    print(f"Saving new index to {new_index_path}...")
    faiss.write_index(new_index, new_index_path)
    
    print(f"Saving new metadata to {new_meta_path}...")
    with open(new_meta_path, "wb") as f:
        pickle.dump(new_meta, f)
        
    t1 = time.time()
    
    # 5. Print summary
    print("\n" + "="*40)
    print("INDEX SHRINK SUMMARY")
    print("="*40)
    print(f"Original chunks     : {len(old_meta)}")
    print(f"New total chunks    : {len(new_meta)}")
    print("New chunks by language:")
    for lang, count in counts.items():
        print(f"  - {lang.ljust(15)}: {count}")
    print(f"New chunks by strategy: {dict(strategy_counts)}")
    
    idx_size_mb = os.path.getsize(new_index_path) / (1024 * 1024)
    meta_size_mb = os.path.getsize(new_meta_path) / (1024 * 1024)
    
    print(f"Final .index size   : {idx_size_mb:.2f} MB")
    print(f"Final .pkl size     : {meta_size_mb:.2f} MB")
    print(f"COMBINED SIZE       : {idx_size_mb + meta_size_mb:.2f} MB")
    
    if (idx_size_mb + meta_size_mb) < 600:
        print("CONFIRMED: Combined size is safely under 600 MB.")
    else:
        print("WARNING: Combined size exceeds 600 MB!")
    
    print(f"Completed in {t1-t0:.2f} seconds.")

if __name__ == "__main__":
    main()
