"""Dense (FAISS), sparse (BM25), hybrid (Reciprocal Rank Fusion) retrieval + cross-encoder reranking."""
import hashlib
import re

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

import config

try:                       # FAISS is optional: a NumPy fallback gives identical (exact) results
    import faiss
except ImportError:
    faiss = None

# BGE models work best with an instruction prefix on the QUERY side only.
QUERY_PREFIX = {
    "BAAI/bge-small-en-v1.5": "Represent this sentence for searching relevant passages: ",
}

_embedders: dict = {}
_reranker = None


def get_embedder(name: str) -> SentenceTransformer:
    if name not in _embedders:
        _embedders[name] = SentenceTransformer(name)
    return _embedders[name]


def get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(config.RERANKER_NAME)
    return _reranker


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


class Index:
    """Holds the chunk list, a FAISS index and a BM25 index."""

    def __init__(self, chunks: list[dict], embed_model: str, show_progress: bool = True):
        self.chunks = chunks
        self.embed_model = embed_model
        self.model = get_embedder(embed_model)
        self.prefix = QUERY_PREFIX.get(embed_model, "")
        texts = [c["text"] for c in chunks]
        emb = self._embed_chunks(texts, show_progress)
        self.emb = emb
        if faiss is not None:
            self.faiss = faiss.IndexFlatIP(emb.shape[1])   # cosine sim (vectors are normalised)
            self.faiss.add(emb)
        else:
            self.faiss = None                               # exact brute-force fallback in dense()
        self.bm25 = BM25Okapi([tokenize(t) for t in texts])

    def _embed_chunks(self, texts: list[str], show_progress: bool):
        """Embed chunks, caching the result on disk (keyed by model + exact chunk texts)."""
        key = hashlib.md5((self.embed_model + "\n" + "\x00".join(texts)).encode("utf-8")).hexdigest()[:16]
        path = config.RESULTS_DIR / "cache" / f"emb_{key}.npy"
        if path.exists():
            return np.load(path)
        emb = self.model.encode(texts, batch_size=32, normalize_embeddings=True,
                                show_progress_bar=show_progress).astype("float32")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, emb)
        return emb

    # ---- individual retrievers: return list of (chunk_index, score) ----
    def dense(self, query: str, n: int):
        q = self.model.encode([self.prefix + query], normalize_embeddings=True).astype("float32")
        n = min(n, len(self.chunks))
        if self.faiss is not None:
            scores, idx = self.faiss.search(q, n)
            return [(int(i), float(s)) for i, s in zip(idx[0], scores[0]) if i >= 0]
        scores = self.emb @ q[0]
        top = np.argsort(-scores)[:n]
        return [(int(i), float(scores[i])) for i in top]

    def sparse(self, query: str, n: int):
        scores = self.bm25.get_scores(tokenize(query))
        top = np.argsort(-scores)[:n]
        return [(int(i), float(scores[i])) for i in top]

    def hybrid(self, query: str, n: int, rrf_k: int = 60):
        """Reciprocal Rank Fusion of dense and BM25 rankings."""
        fused: dict[int, float] = {}
        for ranking in (self.dense(query, n * 2), self.sparse(query, n * 2)):
            for rank, (i, _) in enumerate(ranking):
                fused[i] = fused.get(i, 0.0) + 1.0 / (rrf_k + rank + 1)
        return sorted(fused.items(), key=lambda x: -x[1])[:n]

    # ---- public API ----
    def retrieve(self, query: str, k: int = 5, mode: str = "dense",
                 rerank: bool = False, candidates: int = 20) -> list[dict]:
        n = max(candidates, k) if rerank else k
        if mode == "dense":
            res = self.dense(query, n)
        elif mode == "bm25":
            res = self.sparse(query, n)
        elif mode == "hybrid":
            res = self.hybrid(query, n)
        else:
            raise ValueError(f"unknown mode: {mode}")

        hits = [{**self.chunks[i], "score": s} for i, s in res]
        if rerank and hits:
            scores = get_reranker().predict([(query, h["text"]) for h in hits])
            for h, s in zip(hits, scores):
                h["score"] = float(s)
            hits.sort(key=lambda h: -h["score"])
        return hits[:k]
