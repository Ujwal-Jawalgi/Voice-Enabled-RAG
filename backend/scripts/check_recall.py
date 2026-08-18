import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.pipeline.retrieval import embed_query, search

def main():
    df = pd.read_parquet('backend/data/kanval.parquet')
    row = df[df['query_id'].astype(str) == '1015321'].iloc[0]
    is_selected = list(row['passages']['is_selected'])
    gt_idx = is_selected.index(1) if 1 in is_selected else -1
    gt_passage_id = f"1015321_{gt_idx}"
    query_str = row['query']
    
    print(f"QUERY: {query_str}")
    print(f"GROUND TRUTH: {gt_passage_id}")
    print("-" * 50)
    
    vec = embed_query(query_str)
    results = search(vec, k=5)
    
    for i, res in enumerate(results):
        print(f"[Rank {i+1}] Score: {res.score:.4f} | Lang: {res.language} | ID: {res.passage_id}")
        print(f"Text: {res.text[:100]}...")
        print()

if __name__ == "__main__":
    main()
