import time
import faiss
import pickle
import os
import fsspec
import shutil
import psutil
import pyarrow.parquet as pq
from sentence_transformers import SentenceTransformer
from build_index import chunk_passage, clean_text

MAX_RUNTIME_HOURS = 10
BATCH_SIZE = 256
ROWS_PER_BATCH = 10000
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
LOG_FILE = os.path.join(DATA_DIR, "expand_build_log.txt")
CHECKPOINT_FILE = os.path.join(DATA_DIR, "expand_checkpoint.pkl")
INDEX_PATH = os.path.join(DATA_DIR, "vector_index_expand.faiss")
META_PATH = os.path.join(DATA_DIR, "metadata_expand.pkl")

def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    out = f"[{ts}] {msg}"
    print(out)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(out + "\n")

def phase1_estimate():
    log("=== PHASE 1: ESTIMATE ===")
    fs = fsspec.filesystem("hf")
    
    local_hin = os.path.join(DATA_DIR, "hinval.parquet")
    local_kan = os.path.join(DATA_DIR, "kanval.parquet")
    
    if not os.path.exists(local_hin):
        log("Downloading hinval.parquet...")
        fs.get("datasets/ai4bharat/MSMARCO-XI/validation/hinval.parquet", local_hin)
    if not os.path.exists(local_kan):
        log("Downloading kanval.parquet...")
        fs.get("datasets/ai4bharat/MSMARCO-XI/validation/kanval.parquet", local_kan)
        
    hin_pf = pq.ParquetFile(local_hin)
    kan_pf = pq.ParquetFile(local_kan)
    
    hin_rows = hin_pf.metadata.num_rows
    kan_rows = kan_pf.metadata.num_rows
    
    sample_size = 1000
    hin_passages = 0
    kan_passages = 0
    
    for batch in hin_pf.iter_batches(batch_size=sample_size):
        for row in batch.to_pylist():
            hin_passages += len(row.get("passages", {}).get("English_passages", []))
        break
    
    for batch in kan_pf.iter_batches(batch_size=sample_size):
        for row in batch.to_pylist():
            kan_passages += len(row.get("passages", {}).get("Translated_passages", []))
        break
        
    avg_hin_passages = hin_passages / sample_size
    avg_kan_passages = kan_passages / sample_size
    
    chunk_mult = 1.2
    
    est_eng_chunks = int(hin_rows * avg_hin_passages * chunk_mult)
    est_hin_chunks = int(hin_rows * avg_hin_passages * chunk_mult)
    est_kan_chunks = int(kan_rows * avg_kan_passages * chunk_mult)
    total_est_chunks = est_eng_chunks + est_hin_chunks + est_kan_chunks
    
    speed = 545.0
    est_minutes = total_est_chunks / speed
    
    est_disk_bytes = total_est_chunks * 2300
    est_disk_mb = est_disk_bytes / (1024*1024)
    
    log(f"Row counts: hinval={hin_rows}, kanval={kan_rows}")
    log(f"Passages/row: hinval={avg_hin_passages}, kanval={avg_kan_passages}")
    log(f"Estimated chunks: Eng={est_eng_chunks}, Hin={est_hin_chunks}, Kan={est_kan_chunks}. Total={total_est_chunks}")
    log(f"Projected embedding time: {est_minutes/60:.2f} hours")
    log(f"Projected disk footprint: {est_disk_mb:.2f} MB")
    
    free_space = shutil.disk_usage(DATA_DIR).free
    free_space_mb = free_space / (1024*1024)
    log(f"Available free disk space: {free_space_mb:.2f} MB")
    
    cap = float('inf')
    if est_disk_bytes > 0.7 * free_space:
        safe_bytes = 0.5 * free_space
        safe_chunks = safe_bytes / 2300
        cap = int(safe_chunks / 3)
        log(f"WARNING: Projected footprint exceeds 70% of free space.")
        log(f"FALLBACK TRIGGERED: Capping at {cap} chunks per language to fit safely within 50% free space.")
    else:
        log("Disk space check PASSED.")
        
    return hin_rows, kan_rows, cap

def phase2_build(hin_total, kan_total, cap, script_start_time):
    log("\n=== PHASE 2: CHECKPOINTED BUILD ===")
    
    if os.path.exists(CHECKPOINT_FILE):
        log("Found checkpoint, resuming...")
        with open(CHECKPOINT_FILE, "rb") as f:
            chk = pickle.load(f)
    else:
        log("No checkpoint found, starting fresh...")
        chk = {
            "last_hin_row": 0,
            "last_kan_row": 0,
            "lang_counts": {"english": 0, "hindi": 0, "kannada": 0},
            "seen_texts": set(),
            "metadata_store": [],
            "total_processed_rows": 0
        }
        
    model = SentenceTransformer(MODEL_NAME)
    tokenizer = model.tokenizer
    
    dimension = model.get_sentence_embedding_dimension()
    if os.path.exists(INDEX_PATH):
        log("Loading existing FAISS index...")
        index = faiss.read_index(INDEX_PATH)
    else:
        log("Creating new FAISS index...")
        index = faiss.IndexFlatIP(dimension)
        
    local_hin = os.path.join(DATA_DIR, "hinval.parquet")
    local_kan = os.path.join(DATA_DIR, "kanval.parquet")

    def process_and_commit(chunks, rows_in_batch):
        if not chunks:
            chk["total_processed_rows"] += rows_in_batch
            return True
            
        for attempt in range(4):
            try:
                texts = [c["text"] for c in chunks]
                embeddings = model.encode(texts, batch_size=BATCH_SIZE, normalize_embeddings=True, show_progress_bar=False)
                
                elapsed_hrs = (time.time() - script_start_time) / 3600
                if elapsed_hrs > MAX_RUNTIME_HOURS:
                    log("STOPPED: 10-hour safety limit reached, resume manually.")
                    return False
                    
                index.add(embeddings)
                chk["metadata_store"].extend(chunks)
                chk["total_processed_rows"] += rows_in_batch
                
                faiss.write_index(index, INDEX_PATH)
                with open(META_PATH, "wb") as f:
                    pickle.dump(chk["metadata_store"], f)
                with open(CHECKPOINT_FILE, "wb") as f:
                    pickle.dump(chk, f)
                    
                mem_mb = psutil.Process(os.getpid()).memory_info().rss / (1024*1024)
                log(f"BATCH OK | Processed Rows: {chk['total_processed_rows']} | Langs: {chk['lang_counts']} | Mem: {mem_mb:.1f}MB | Elapsed: {elapsed_hrs:.1f}h")
                return True
                
            except Exception as e:
                log(f"Batch failed (attempt {attempt+1}/4): {e}")
                if attempt < 3:
                    time.sleep(30)
                else:
                    log(f"SKIPPING BATCH due to repeated failures.")
                    chk["total_processed_rows"] += rows_in_batch
                    with open(CHECKPOINT_FILE, "wb") as f:
                        pickle.dump(chk, f)
                    return True

    # HINDI + ENGLISH
    log("Processing Hindi & English...")
    hin_pf = pq.ParquetFile(local_hin)
    row_idx = 0
    batch_chunks = []
    rows_in_batch = 0
    
    for batch in hin_pf.iter_batches():
        for row in batch.to_pylist():
            row_idx += 1
            if row_idx <= chk["last_hin_row"]:
                continue
                
            if chk["lang_counts"]["english"] < cap or chk["lang_counts"]["hindi"] < cap:
                query_id = str(row.get("query_id", ""))
                passages = row.get("passages", {})
                if passages:
                    eng_p = passages.get("English_passages", [])
                    trans_p = passages.get("Translated_passages", [])
                    sel = passages.get("is_selected", [])
                    for p_idx, (ep, tp, s) in enumerate(zip(eng_p, trans_p, sel)):
                        if chk["lang_counts"]["english"] < cap:
                            ce = clean_text(ep)
                            if ce and ce not in chk["seen_texts"]:
                                chk["seen_texts"].add(ce)
                                chks = chunk_passage(ce, f"{query_id}_{p_idx}", "english", query_id, s, tokenizer)
                                batch_chunks.extend(chks)
                                chk["lang_counts"]["english"] += len(chks)
                        if chk["lang_counts"]["hindi"] < cap:
                            ch = clean_text(tp)
                            if ch and ch not in chk["seen_texts"]:
                                chk["seen_texts"].add(ch)
                                chks = chunk_passage(ch, f"{query_id}_{p_idx}", "hindi", query_id, s, tokenizer)
                                batch_chunks.extend(chks)
                                chk["lang_counts"]["hindi"] += len(chks)
                                
            rows_in_batch += 1
            chk["last_hin_row"] = row_idx
            
            if rows_in_batch >= ROWS_PER_BATCH:
                if not process_and_commit(batch_chunks, rows_in_batch):
                    return
                batch_chunks = []
                rows_in_batch = 0
                
    if rows_in_batch > 0:
        if not process_and_commit(batch_chunks, rows_in_batch):
            return

    # KANNADA
    log("Processing Kannada...")
    kan_pf = pq.ParquetFile(local_kan)
    row_idx = 0
    batch_chunks = []
    rows_in_batch = 0
    
    for batch in kan_pf.iter_batches():
        for row in batch.to_pylist():
            row_idx += 1
            if row_idx <= chk["last_kan_row"]:
                continue
                
            if chk["lang_counts"]["kannada"] < cap:
                query_id = str(row.get("query_id", ""))
                passages = row.get("passages", {})
                if passages:
                    trans_p = passages.get("Translated_passages", [])
                    sel = passages.get("is_selected", [])
                    for p_idx, (tp, s) in enumerate(zip(trans_p, sel)):
                        ck = clean_text(tp)
                        if ck and ck not in chk["seen_texts"]:
                            chk["seen_texts"].add(ck)
                            chks = chunk_passage(ck, f"{query_id}_{p_idx}", "kannada", query_id, s, tokenizer)
                            batch_chunks.extend(chks)
                            chk["lang_counts"]["kannada"] += len(chks)
                            
            rows_in_batch += 1
            chk["last_kan_row"] = row_idx
            
            if rows_in_batch >= ROWS_PER_BATCH:
                if not process_and_commit(batch_chunks, rows_in_batch):
                    return
                batch_chunks = []
                rows_in_batch = 0
                
    if rows_in_batch > 0:
        process_and_commit(batch_chunks, rows_in_batch)

    log("Phase 2 Complete!")

def main():
    start_time = time.time()
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("=== EXPAND BUILD PIPELINE INITIATED ===\n")
        
    try:
        hin_total, kan_total, cap = phase1_estimate()
        phase2_build(hin_total, kan_total, cap, start_time)
        
        log("\n=== PHASE 3: MMAP VERIFICATION ===")
        if os.path.exists(INDEX_PATH):
            index_size = os.path.getsize(INDEX_PATH) / (1024*1024)
            meta_size = os.path.getsize(META_PATH) / (1024*1024)
            log(f"Final Artifacts - FAISS: {index_size:.1f}MB, Meta: {meta_size:.1f}MB")
            log("Verifying MMAP load...")
            idx = faiss.read_index(INDEX_PATH, faiss.IO_FLAG_MMAP)
            log(f"Loaded successfully with MMAP! Vectors: {idx.ntotal}")
    except Exception as e:
        log(f"CRITICAL ERROR IN PIPELINE: {e}")

if __name__ == "__main__":
    main()
