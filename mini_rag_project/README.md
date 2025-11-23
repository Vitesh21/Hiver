# Mini RAG Project

A modular retrieval-augmented generation (RAG) implementation with pluggable embedder, retriever, and generator.

## Structure

- `src/embedder.py` — Sentence-transformer embeddings
- `src/retriever.py` — FAISS (with NumPy fallback)
- `src/generator.py` — Extractive answer generator
- `src/rag_pipeline.py` — Main pipeline
- `outputs/results.json` — Saved run results

## Setup

1. Create a virtual environment.
2. Install dependencies: `pip install -r requirements.txt` (use project root or this folder).

## Run

Basic:
```
python -m mini_rag_project.src.rag_pipeline
```

With options (uses top-level `kb` by default):
```
python -m mini_rag_project.src.rag_pipeline --kb_dir kb --chunk_size 120 --chunk_overlap 30 --top_k 3
```

Notes:
- `--kb_dir` can be relative to project root (e.g., `kb`) or absolute; default is the project root `kb`.
- `--chunk_size` and `--chunk_overlap` are word-based.
- Outputs are saved to `mini_rag_project/outputs/results.json`.

## Improve Retrieval

- Chunk articles into 200–400 tokens.
- Use a stronger embedding model (e.g., `all-mpnet-base-v2`).
- Hybrid retrieval (dense + BM25).
- Re-rank with a local LLM on top-k.
- Add metadata filters (area, category, type).

## Troubleshooting
- If FAISS is unavailable, the retriever falls back to NumPy distance.
- For very small KBs, similar chunks from the same doc may repeat in top-k; chunking and more documents improve diversity.