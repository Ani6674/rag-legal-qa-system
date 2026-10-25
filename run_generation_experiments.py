"""Compare no-RAG baseline vs RAG variants end-to-end, for one LLM per run.

Examples:
  python run_generation_experiments.py --llm ollama --judge-llm anthropic
  python run_generation_experiments.py --llm anthropic --judge-llm anthropic --limit 20

Run it once per LLM you want to compare (e.g. small open model vs larger API model).
Use a STRONGER model as judge than the one being evaluated, and say so in the report.

Output: results/generation_<llm>.csv (per question) and generation_summary_<llm>.csv
"""
import argparse
import json

import pandas as pd

import config
from ingest import chunk_documents, load_documents
from llm import LLM
from metrics import (CORRECTNESS_PROMPT, FAITHFULNESS_PROMPT, bert_score_f1, judge,
                     retrieval_scores, rouge_l)
from rag import RAGPipeline
from retriever import Index

# name -> (use_rag, retrieval mode, rerank)
VARIANTS = {
    "no_rag": (False, None, False),
    "rag_dense": (True, "dense", False),
    "rag_hybrid": (True, "hybrid", False),
    "rag_hybrid_rerank": (True, "hybrid", True),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", default=None)
    ap.add_argument("--judge-llm", default=None)
    ap.add_argument("--chunk-size", type=int, default=config.DEFAULT_CHUNK_SIZE)
    ap.add_argument("--overlap", type=int, default=config.DEFAULT_OVERLAP)
    ap.add_argument("--embedder", default="bge-small", choices=list(config.EMBEDDERS))
    ap.add_argument("--k", type=int, default=config.DEFAULT_K)
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-bertscore", action="store_true")
    ap.add_argument("--no-judge", action="store_true",
                    help="skip LLM-as-judge scores (faster; use when the judge model is too weak)")
    ap.add_argument("--max-tokens", type=int, default=150, help="max answer length (shorter = faster)")
    args = ap.parse_args()

    tests = json.loads(config.TESTSET_PATH.read_text(encoding="utf-8"))[: args.limit]
    chunks = chunk_documents(load_documents(config.RAW_DIR), args.chunk_size, args.overlap)
    index = Index(chunks, config.EMBEDDERS[args.embedder])
    llm, judge_llm = LLM(args.llm), LLM(args.judge_llm or args.llm)
    pipe = RAGPipeline(index, llm)

    rows = []
    for vname in args.variants:
        use_rag, mode, rerank = VARIANTS[vname]
        print(f"\n== {vname} | LLM={llm.name} ==")
        for i, t in enumerate(tests, 1):
            r = pipe.answer(t["question"], k=args.k, mode=mode or "dense", rerank=rerank, use_rag=use_rag,
                            max_tokens=args.max_tokens)
            ctx = "\n\n".join(r["contexts"])
            row = {
                "variant": vname, "llm": llm.name, "judge": judge_llm.name,
                "question": t["question"], "reference": t["reference_answer"], "answer": r["answer"],
                "rouge_l": rouge_l(r["answer"], t["reference_answer"]),
                "correctness": None if args.no_judge else judge(judge_llm, CORRECTNESS_PROMPT.format(
                    question=t["question"], reference=t["reference_answer"], answer=r["answer"])),
                "faithfulness": judge(judge_llm, FAITHFULNESS_PROMPT.format(
                    context=ctx, answer=r["answer"])) if (use_rag and not args.no_judge) else None,
                "manual_score": "",   # fill in by hand in Excel: 0 = wrong, 1 = partly right, 2 = correct
                "context_hit": retrieval_scores(r["contexts"], t["evidence"], args.k)["hit"] if use_rag else None,
                "retrieval_s": r["retrieval_s"], "generation_s": r["generation_s"], "total_s": r["total_s"],
            }
            rows.append(row)
            pd.DataFrame(rows).to_csv(config.RESULTS_DIR / "generation_partial.csv", index=False)
            print(f"  [{i}/{len(tests)}] correctness={row['correctness']} faithfulness={row['faithfulness']}")

    df = pd.DataFrame(rows)
    if not args.no_bertscore:
        df["bertscore_f1"] = bert_score_f1(df["answer"].tolist(), df["reference"].tolist())

    for col in ("correctness", "faithfulness", "context_hit"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    tag = llm.name.replace("/", "_").replace(":", "_")
    df.to_csv(config.RESULTS_DIR / f"generation_{tag}.csv", index=False)
    num = [c for c in ["rouge_l", "bertscore_f1", "correctness", "faithfulness", "context_hit",
                       "retrieval_s", "generation_s", "total_s"] if c in df.columns]
    summary = df.groupby("variant")[num].mean().round(3).reindex(args.variants)
    summary.to_csv(config.RESULTS_DIR / f"generation_summary_{tag}.csv")
    print("\n", summary.to_string())


if __name__ == "__main__":
    main()
