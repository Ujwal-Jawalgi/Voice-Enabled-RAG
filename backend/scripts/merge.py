import faiss, pickle, numpy as np, sys

sys.stdout.reconfigure(encoding='utf-8')
# Quick inspection first
with open("metadata.pkl", "rb") as f:
    meta_a = pickle.load(f)
with open("metadata_12langs.pkl", "rb") as f:
    meta_b = pickle.load(f)

print("Sample from 3-lang metadata:", meta_a[0])
print("Sample from 12-lang metadata:", meta_b[0])

# Normalize types so both metadata sets match
for chunk in meta_b:
    chunk["source_query_id"] = str(chunk["source_query_id"])
    chunk["is_selected"] = int(chunk["is_selected"])

for chunk in meta_a:
    chunk["source_query_id"] = str(chunk["source_query_id"])
    chunk["is_selected"] = int(chunk["is_selected"])

# Merge
index_a = faiss.read_index("vector_index.faiss")
index_b = faiss.read_index("faiss_index_12langs.index")

merged_vectors = np.vstack([
    index_a.reconstruct_n(0, index_a.ntotal),
    index_b.reconstruct_n(0, index_b.ntotal)
])
merged_index = faiss.IndexFlatIP(merged_vectors.shape[1])
merged_index.add(merged_vectors)
merged_metadata = meta_a + meta_b

faiss.write_index(merged_index, "faiss_index_full.index")
with open("metadata_full.pkl", "wb") as f:
    pickle.dump(merged_metadata, f)

print(f"Merged index total: {merged_index.ntotal} vectors")
