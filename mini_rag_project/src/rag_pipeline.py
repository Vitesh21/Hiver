import os
import json
import argparse
import numpy as np

from .embedder import Embedder
from .retriever import Retriever
from .generator import generate_answer  # or llama_answer
from .utils import project_root_from_here, ensure_dir


def resolve_kb_dir(folder: str | None) -> str:
    this_file = os.path.abspath(__file__)
    pkg_root = project_root_from_here(this_file)  # mini_rag_project
    workspace_root = os.path.dirname(pkg_root)    # project root (lyz)
    if folder is None:
        return os.path.join(workspace_root, "kb")
    # If absolute, use as-is; else resolve relative to workspace root
    return folder if os.path.isabs(folder) else os.path.join(workspace_root, folder)


def load_articles(folder: str | None = None):
    kb_dir = resolve_kb_dir(folder)
    articles = []
    ids = []
    for f in os.listdir(kb_dir):
        if f.endswith(".md") or f.endswith(".txt"):
            with open(os.path.join(kb_dir, f), "r", encoding="utf-8") as file:
                articles.append(file.read())
                ids.append(f)
    return ids, articles


def chunk_text(text: str, size: int = 120, overlap: int = 30) -> list[str]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(len(words), start + size)
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end == len(words):
            break
        start = max(end - overlap, start + 1)
    return chunks


def run_rag(kb_dir: str | None = None, chunk_size: int = 120, chunk_overlap: int = 30, top_k: int = 3):
    ids, articles = load_articles(folder=kb_dir)
    # Build chunks and metadata
    chunk_texts: list[str] = []
    chunk_meta: list[dict] = []
    for doc_id, article in zip(ids, articles):
        chunks = chunk_text(article, size=chunk_size, overlap=chunk_overlap)
        for i, ch in enumerate(chunks):
            chunk_texts.append(ch)
            chunk_meta.append({"doc_id": doc_id, "chunk_id": i})

    embedder = Embedder()
    chunk_embeddings = embedder.embed(chunk_texts)
    retriever = Retriever(chunk_embeddings)

    queries = [
        "How do I configure automations in Hiver?",
        "Why is CSAT not appearing?",
    ]

    results = {}

    for query in queries:
        q_emb = embedder.embed([query])[0]
        idxs, sims = retriever.search(q_emb, top_k=top_k)

        retrieved_texts = [chunk_texts[i] for i in idxs]
        retrieved_meta = [chunk_meta[i] for i in idxs]

        answer = generate_answer(query, retrieved_texts)
        confidence = float(np.mean(sims))

        results[query] = {
            "retrieved": [
                {
                    "doc_id": retrieved_meta[i]["doc_id"],
                    "chunk_id": retrieved_meta[i]["chunk_id"],
                    "similarity": float(sims[i]),
                }
                for i in range(len(retrieved_texts))
            ],
            "generated_answer": answer,
            "confidence_score": confidence,
        }

    # Save results to outputs
    this_file = os.path.abspath(__file__)
    root = project_root_from_here(this_file)
    out_dir = os.path.join(root, "outputs")
    ensure_dir(out_dir)
    out_path = os.path.join(out_dir, "results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    print(f"RAG Completed. Output saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mini RAG pipeline")
    parser.add_argument("--kb_dir", type=str, default=None, help="KB directory relative to project root or absolute path")
    parser.add_argument("--chunk_size", type=int, default=120, help="Chunk size in words")
    parser.add_argument("--chunk_overlap", type=int, default=30, help="Chunk overlap in words")
    parser.add_argument("--top_k", type=int, default=3, help="Top K chunks to retrieve")
    args = parser.parse_args()

    run_rag(kb_dir=args.kb_dir, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap, top_k=args.top_k)