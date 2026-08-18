# Chunking and Subsampling Strategy

## Subsampling Decision
The full Indic MSMARCO-XI dataset contains over 11.45 million passages across 14 languages (plus English). Processing, embedding, and hosting this entire dataset locally is infeasible within a 4-day hackathon deadline and hardware constraints.

**Decision**: We strategically subsampled the dataset to cover all 15 languages at an asymmetric scale:
- **English**: 150,000 passages
- **Hindi**: 150,000 passages
- **Kannada**: 150,000 passages
- **12 remaining languages**: ~20,000 passages each (240,000 total)

This yields a dense, multilingual corpus of **~690,000 passages**. The resulting 1.06 GB FAISS index fits entirely in RAM alongside the `paraphrase-multilingual-MiniLM-L12-v2` embedding model. This allows us to achieve local exact-search retrieval (Embedding + FAISS + BM25 RRF) in ~230ms on standard CPU hardware.

## Chunking Strategy
Due to the context window limitations of dense embedding models, passages that are too long must be split.

1. **Primary Strategy (`passage`)**:
   - We use the `SentenceTransformer` tokenizer to encode the text.
   - If the passage is <= 256 tokens, it is kept intact as a single chunk.

2. **Fallback Strategy (`fixed_overlap`)**:
   - If the passage exceeds 256 tokens, we use a sliding window approach.
   - **Chunk Size**: 256 tokens
   - **Overlap**: 32 tokens
   - Each resulting chunk retains the original `passage_id` appended with `_partN` to ensure provenance.

This hybrid approach ensures that shorter passages remain coherent single blocks, while longer documents are split securely without losing critical context at the boundaries.
