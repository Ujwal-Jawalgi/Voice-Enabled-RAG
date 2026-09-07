import sys
import os

# Add backend directory to sys.path so we can import app
backend_dir = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, backend_dir)

from app.pipeline.retrieval import embed_query, search

import sys
sys.stdout.reconfigure(encoding='utf-8')

def main():
    queries = [
        # Indo-Aryan
        ("What are the symptoms of Covid-19?", "english"),
        ("कोविड-19 के लक्षण क्या हैं?", "hindi"), # Hindi
        ("কোভিড-১৯ এর লক্ষণ কী কী?", "bengali"), # Bengali
        # Dravidian
        ("ಕೋವಿಡ್-19 ನ ಲಕ್ಷಣಗಳು ಯಾವುವು?", "kannada"), # Kannada
        ("കോവിഡ് -19 ന്റെ ലക്ഷണങ്ങൾ എന്തൊക്കെയാണ്?", "malayalam") # Malayalam
    ]
    
    print("Running sanity checks against new FAISS index...")
    for q_text, lang in queries:
        print(f"\n--- Query: '{q_text}' ({lang}) ---")
        vec = embed_query(q_text)
        results = search(vec, k=3)
        for i, r in enumerate(results):
            print(f"Rank {i+1} [Score: {r.score:.4f} | Lang: {r.language}]")
            print(f"Text: {r.text[:150]}...")

if __name__ == "__main__":
    main()
