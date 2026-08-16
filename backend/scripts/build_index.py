import time
import re
import unicodedata
import faiss
import pickle
import os
import fsspec
import pyarrow.parquet as pq
from sentence_transformers import SentenceTransformer

MAX_CHUNKS_PER_LANG = 6000
CHAR_LENGTH_THRESHOLD = 1000
TOKEN_MAX_SIZE = 100
TOKEN_OVERLAP = 20
BATCH_SIZE = 256
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
DATA_DIR = os.environ.get("RAILWAY_VOLUME_MOUNT_PATH", DEFAULT_DATA_DIR)

def clean_text(text: str) -> str:
    if not text:
        return ""
    # Strip HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Normalize unicode
    text = unicodedata.normalize('NFKC', text)
    # Collapse whitespaces
    text = ' '.join(text.split())
    return text

def chunk_passage(text: str, passage_id: str, lang: str, source_query_id: str, is_selected: int, tokenizer):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) <= TOKEN_MAX_SIZE:
        return [{
            "text": text,
            "passage_id": passage_id,
            "language": lang,
            "source_query_id": source_query_id,
            "is_selected": is_selected,
            "char_length": len(text),
            "chunk_strategy": "passage"
        }]
        
    chunks = []
    stride = TOKEN_MAX_SIZE - TOKEN_OVERLAP
    for i in range(0, len(tokens), stride):
        chunk_tokens = tokens[i:i + TOKEN_MAX_SIZE]
        chunk_text = tokenizer.decode(chunk_tokens)
        chunks.append({
            "text": chunk_text,
            "passage_id": f"{passage_id}_part{i//stride}",
            "language": lang,
            "source_query_id": source_query_id,
            "is_selected": is_selected,
            "char_length": len(chunk_text),
            "chunk_strategy": "fixed_overlap"
        })
    return chunks

def build_index():
    t0 = time.perf_counter()
    os.makedirs(DATA_DIR, exist_ok=True)
    
    print(f"Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    model.max_seq_length = 128
    tokenizer = model.tokenizer
    
    fs = fsspec.filesystem("hf")
    
    chunks_store = []
    seen_texts = set()
    lang_counts = {"english": 0, "hindi": 0, "kannada": 0}
    strategy_counts = {"passage": 0, "fixed_overlap": 0}
    
    # --- Process Hindi & English from hinval.parquet ---
    print("Extracting Hindi and English from hinval.parquet...")
    with fs.open("datasets/ai4bharat/MSMARCO-XI/validation/hinval.parquet", "rb") as f:
        pf = pq.ParquetFile(f)
        for batch in pf.iter_batches():
            if lang_counts["hindi"] >= MAX_CHUNKS_PER_LANG and lang_counts["english"] >= MAX_CHUNKS_PER_LANG:
                break
                
            for row in batch.to_pylist():
                if lang_counts["hindi"] >= MAX_CHUNKS_PER_LANG and lang_counts["english"] >= MAX_CHUNKS_PER_LANG:
                    break
                    
                query_id = str(row.get("query_id", ""))
                passages = row.get("passages", {})
                if not passages:
                    continue
                    
                eng_passages = passages.get("English_passages", [])
                trans_passages = passages.get("Translated_passages", [])
                is_selected = passages.get("is_selected", [])
                
                # Zip through parallel passage lists
                for p_idx, (eng_p, trans_p, sel) in enumerate(zip(eng_passages, trans_passages, is_selected)):
                    # English chunks
                    if lang_counts["english"] < MAX_CHUNKS_PER_LANG:
                        clean_eng = clean_text(eng_p)
                        if clean_eng and clean_eng not in seen_texts:
                            seen_texts.add(clean_eng)
                            pid = f"{query_id}_{p_idx}"
                            new_chunks = chunk_passage(clean_eng, pid, "english", query_id, sel, tokenizer)
                            for c in new_chunks:
                                chunks_store.append(c)
                                lang_counts["english"] += 1
                                strategy_counts[c["chunk_strategy"]] += 1
                                if lang_counts["english"] >= MAX_CHUNKS_PER_LANG:
                                    break
                                    
                    # Hindi chunks
                    if lang_counts["hindi"] < MAX_CHUNKS_PER_LANG:
                        clean_hin = clean_text(trans_p)
                        if clean_hin and clean_hin not in seen_texts:
                            seen_texts.add(clean_hin)
                            pid = f"{query_id}_{p_idx}"
                            new_chunks = chunk_passage(clean_hin, pid, "hindi", query_id, sel, tokenizer)
                            for c in new_chunks:
                                chunks_store.append(c)
                                lang_counts["hindi"] += 1
                                strategy_counts[c["chunk_strategy"]] += 1
                                if lang_counts["hindi"] >= MAX_CHUNKS_PER_LANG:
                                    break

    # --- Process Kannada from kanval.parquet ---
    print("Extracting Kannada from kanval.parquet...")
    with fs.open("datasets/ai4bharat/MSMARCO-XI/validation/kanval.parquet", "rb") as f:
        pf = pq.ParquetFile(f)
        for batch in pf.iter_batches():
            if lang_counts["kannada"] >= MAX_CHUNKS_PER_LANG:
                break
                
            for row in batch.to_pylist():
                if lang_counts["kannada"] >= MAX_CHUNKS_PER_LANG:
                    break
                    
                query_id = str(row.get("query_id", ""))
                passages = row.get("passages", {})
                if not passages:
                    continue
                    
                trans_passages = passages.get("Translated_passages", [])
                is_selected = passages.get("is_selected", [])
                
                for p_idx, (trans_p, sel) in enumerate(zip(trans_passages, is_selected)):
                    if lang_counts["kannada"] < MAX_CHUNKS_PER_LANG:
                        clean_kan = clean_text(trans_p)
                        if clean_kan and clean_kan not in seen_texts:
                            seen_texts.add(clean_kan)
                            pid = f"{query_id}_{p_idx}"
                            new_chunks = chunk_passage(clean_kan, pid, "kannada", query_id, sel, tokenizer)
                            for c in new_chunks:
                                chunks_store.append(c)
                                lang_counts["kannada"] += 1
                                strategy_counts[c["chunk_strategy"]] += 1
                                if lang_counts["kannada"] >= MAX_CHUNKS_PER_LANG:
                                    break

    t_extract = time.perf_counter()
    print(f"Extraction complete in {t_extract - t0:.2f}s. Total chunks: {len(chunks_store)}")
    
    # --- Batch Embedding ---
    print(f"Generating embeddings for {len(chunks_store)} chunks...")
    texts = [c["text"] for c in chunks_store]
    embeddings = model.encode(texts, batch_size=BATCH_SIZE, normalize_embeddings=True, show_progress_bar=True)
    
    # --- Build FAISS Index ---
    print("Building FAISS IndexFlatIP...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    
    # --- Save Artifacts ---
    index_path = os.path.join(DATA_DIR, "vector_index.faiss")
    meta_path = os.path.join(DATA_DIR, "metadata.pkl")
    faiss.write_index(index, index_path)
    
    # Strip full text from metadata to save memory during inference if desired, 
    # but we need text to return the actual passages for grounding in the harness.
    with open(meta_path, "wb") as f:
        pickle.dump(chunks_store, f)
        
    t_end = time.perf_counter()
    
    # --- Final Summary ---
    print("\n" + "="*40)
    print("BUILD INDEX SUMMARY")
    print("="*40)
    print(f"Total chunks          : {len(chunks_store)}")
    print(f"Chunks by language    : {lang_counts}")
    print(f"Chunks by strategy    : {strategy_counts}")
    print(f"Total build time      : {t_end - t0:.2f}s")
    print(f"Index saved to        : {index_path}")
    print(f"Metadata saved to     : {meta_path}")

if __name__ == "__main__":
    build_index()
