import os
import pickle
import faiss
import numpy as np
from collections import Counter
import time

def main():
    t0 = time.time()
    
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    
    # We read from the original massive backup to ensure we pull fresh original data for English, Hindi, Kannada
    old_index_path = os.path.join(data_dir, "vector_index.faiss.bak")
    old_meta_path = os.path.join(data_dir, "metadata.pkl.bak")
    
    new_index_path = os.path.join(data_dir, "vector_index.faiss")
    new_meta_path = os.path.join(data_dir, "metadata.pkl")
    
    if not os.path.exists(old_index_path) or not os.path.exists(old_meta_path):
        print(f"Error: Original .bak files not found in {data_dir}. Cannot proceed.")
        return

    # 1. Load existing massive DB
    print(f"Loading original massive FAISS index from {old_index_path}...")
    old_index = faiss.read_index(old_index_path)
    
    print(f"Loading original massive metadata from {old_meta_path}...")
    with open(old_meta_path, "rb") as f:
        old_meta = pickle.load(f)
        
    assert len(old_meta) == old_index.ntotal, "Metadata length does not match index size!"
    
    # 2. Extract and filter
    target_languages = {"english", "hindi", "kannada"}
    max_per_language = 3300
    
    counts = Counter()
    
    new_meta = []
    new_vectors = []
    
    print(f"Filtering rows for Render deployment (max {max_per_language} per language: {target_languages})...")
    for i in range(old_index.ntotal):
        meta = old_meta[i]
        lang = meta["language"]
        
        if lang in target_languages and counts[lang] < max_per_language:
            new_meta.append(meta)
            new_vectors.append(old_index.reconstruct(i))
            counts[lang] += 1
            
    # 3. Build new index
    if len(new_vectors) == 0:
        print("Error: No vectors found for target languages.")
        return
        
    new_vectors_np = np.array(new_vectors, dtype=np.float32)
    dimension = new_vectors_np.shape[1]
    
    print("Building microscopic FAISS index...")
    new_index = faiss.IndexFlatIP(dimension)
    new_index.add(new_vectors_np)
    
    # 4. Save
    print(f"Saving tiny index to {new_index_path}...")
    faiss.write_index(new_index, new_index_path)
    
    print(f"Saving tiny metadata to {new_meta_path}...")
    with open(new_meta_path, "wb") as f:
        pickle.dump(new_meta, f)
        
    t1 = time.time()
    
    # 5. Print summary
    print("\n" + "="*40)
    print("RENDER INDEX SUMMARY")
    print("="*40)
    print(f"Original chunks     : {len(old_meta)}")
    print(f"New total chunks    : {len(new_meta)}")
    print("New chunks by language:")
    for lang, count in counts.items():
        print(f"  - {lang.ljust(15)}: {count}")
    
    idx_size_mb = os.path.getsize(new_index_path) / (1024 * 1024)
    meta_size_mb = os.path.getsize(new_meta_path) / (1024 * 1024)
    
    print(f"Final .index size   : {idx_size_mb:.2f} MB")
    print(f"Final .pkl size     : {meta_size_mb:.2f} MB")
    print(f"COMBINED SIZE       : {idx_size_mb + meta_size_mb:.2f} MB")
    
    if (idx_size_mb + meta_size_mb) < 200:
        print("CONFIRMED: Combined size is safely under Render's memory limits.")
    else:
        print("WARNING: Combined size might be too large for Render!")
    
    print(f"Completed in {t1-t0:.2f} seconds.")

if __name__ == "__main__":
    main()
