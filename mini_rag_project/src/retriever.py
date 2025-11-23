import numpy as np


class Retriever:
    def __init__(self, embeddings: np.ndarray):
        self.embeddings = embeddings.astype(np.float32)
        self.dim = embeddings.shape[1]
        # Try to use FAISS if available; otherwise fallback to NumPy
        try:
            import faiss  # type: ignore
            self.faiss = faiss
            self.index = faiss.IndexFlatL2(self.dim)
            self.index.add(self.embeddings)
        except Exception:
            self.faiss = None
            self.index = None

    def search(self, query_embedding: np.ndarray, top_k: int = 3):
        q = query_embedding.reshape(1, -1).astype(np.float32)
        if self.faiss is not None and self.index is not None:
            distances, idxs = self.index.search(q, top_k)
            sims = 1.0 / (1.0 + distances[0])
            return idxs[0], sims
        # Fallback: compute L2 distances with NumPy
        diffs = self.embeddings - q
        distances = np.sum(diffs * diffs, axis=1)
        idxs = np.argsort(distances)[:top_k]
        sims = 1.0 / (1.0 + distances[idxs])
        return idxs, sims