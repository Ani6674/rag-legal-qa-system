"""Grid search over chunk size x embedder x retrieval mode x reranking x k. No LLM needed (fast).

Usage:  python run_retrieval_experiments.py
Output: results/retrieval_results.csv
"""
import argparse
import json
import time

import pandas as pd

import config
from ingest import chunk_documents, load_documents
from metrics import retrieval_scores
from retriever import Index

# (chunk_size_words, overlap_words). NOTE: MiniLM truncates input at ~256 word-pieces
# (~190 words), so very large chunks lose their tail for that model - worth discussing.
CHUNK_CONFIGS = [(100, 20), (200, 40), (400, 80)]
MODES = ["dense", "bm25", "hybrid"]
KS = [3, 5, 10]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedders", nargs="+", default=list(config.EMBEDDERS))
    args = ap.parse_args()

    tests = json.loads(config.TESTSET_PATH.read_text(encoding="utf-8"))
    docs = load_documents(config.RAW_DIR)
    rows = []
    print(f"{len(tests)} questions, {len(docs)} pages")

    for cs, ov in CHUNK_CONFIGS:
        chunks = chunk_documents(docs, cs, ov)
        for ei, ename in enumerate(args.embedders):
            index = Index(chunks, config.EMBEDDERS[ename])
            for mode in MODES:
                if mode == "bm25" and ei > 0:       # BM25 is embedder-independent
                    continue
                label = "none" if mode == "bm25" else ename
                for rerank in (False, True):
                    per_q, lat = [], []
                    for t in tests:
                        t0 = time.perf_counter()
                        hits = index.retrieve(t["question"], k=max(KS), mode=mode, rerank=rerank)
                        lat.append((time.perf_counter() - t0) * 1000)
                        per_q.append([h["text"] for h in hits])
                    for k in KS:
                        sc = [retrieval_scores(texts, t["evidence"], k) for texts, t in zip(per_q, tests)]
                        rows.append({
                            "chunk_size": cs, "overlap": ov, "n_chunks": len(chunks),
                            "embedder": label, "mode": mode, "rerank": rerank, "k": k,
                            "hit_rate": sum(s["hit"] for s in sc) / len(sc),
                            "recall": sum(s["recall"] for s in sc) / len(sc),
                            "mrr": sum(s["rr"] for s in sc) / len(sc),
                            "latency_ms": sum(lat) / len(lat),
                        })
                    print(f"cs={cs} emb={label} mode={mode} rerank={rerank} "
                          f"hit@5={rows[-2]['hit_rate']:.2f} mrr@5={rows[-2]['mrr']:.2f}")

    out = config.RESULTS_DIR / "retrieval_results.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
