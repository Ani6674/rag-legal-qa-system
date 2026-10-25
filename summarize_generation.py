"""Summarise manual grading (and other metrics) from the generation results CSV.

Usage:  python summarize_generation.py                 (uses the newest results/generation_*.csv)
        python summarize_generation.py results/generation_llama3.2_3b.csv

Needs the `manual_score` column filled in (0 = wrong, 1 = partly right, 2 = correct)
and the file saved as CSV UTF-8.
"""
import argparse
import glob
import io
import os
import sys

import pandas as pd

import config

ap = argparse.ArgumentParser()
ap.add_argument("file", nargs="?", help="results CSV (default: newest results/generation_*.csv)")
ap.add_argument("--score-col", default="manual_score", help="column holding the 0/1/2 scores")
args = ap.parse_args()

if args.file:
    path = args.file
else:
    files = [f for f in glob.glob(str(config.RESULTS_DIR / "generation_*.csv"))
             if "summary" not in f and "partial" not in f]
    if not files:
        sys.exit("No generation_*.csv found in results/")
    path = max(files, key=os.path.getmtime)

df = pd.read_csv(path)
df["manual_score"] = pd.to_numeric(df[args.score_col], errors="coerce")
df["abstained"] = df["answer"].fillna("").str.contains("could not find", case=False)
missing = int(df["manual_score"].isna().sum())
if missing == len(df):
    sys.exit(f"{args.score_col} is empty. Fill it in (0/1/2), save as CSV UTF-8, and run again.")

buf = io.StringIO()
buf.write(f"File: {path}\nRows without a manual score: {missing} of {len(df)}\n")
sc = df.dropna(subset=["manual_score"])

summ = sc.groupby("variant").agg(
    n=("manual_score", "size"),
    mean_score_0to2=("manual_score", "mean"),
    pct_correct=("manual_score", lambda s: 100 * (s == 2).mean()),
    pct_partly_or_better=("manual_score", lambda s: 100 * (s >= 1).mean()),
    pct_wrong=("manual_score", lambda s: 100 * (s == 0).mean()),
    rouge_l=("rouge_l", "mean"),
    pct_abstained=("abstained", lambda s: 100 * s.mean()),
    total_s=("total_s", "mean"),
)
buf.write(f"\n=== Manual grading by variant ===\n{summ.round(2).to_string()}\n")

piv = sc.pivot_table(index="question", columns="variant", values="manual_score")
if "no_rag" in piv:
    for v in [c for c in piv.columns if c != "no_rag"]:
        both = piv[["no_rag", v]].dropna()
        better, worse, same = (both[v] > both["no_rag"]).sum(), (both[v] < both["no_rag"]).sum(), (both[v] == both["no_rag"]).sum()
        buf.write(f"\n=== Paired comparison on {len(both)} questions: {v} vs no_rag ===\n"
                  f"{v} better: {better} | worse: {worse} | same: {same}\n")

rag = sc[(sc["variant"] != "no_rag") & sc["context_hit"].notna()]
if not rag.empty:
    by_hit = rag.groupby(["variant", "context_hit"])["manual_score"].agg(n="size", mean_score="mean")
    buf.write("\n=== RAG answer quality when the right passage was / was not retrieved ===\n"
              f"(context_hit 1 = evidence retrieved, 0 = missed)\n{by_hit.round(2).to_string()}\n")

print(buf.getvalue())
(config.RESULTS_DIR / "generation_manual_summary.txt").write_text(buf.getvalue(), encoding="utf-8")
print("Saved results/generation_manual_summary.txt")
