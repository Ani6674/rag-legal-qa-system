"""Spot-check a sample of the draft scores yourself.

Usage:  python spot_check.py
        python spot_check.py results/generation_llama3.2_3b_draft_scored.csv --n 15

What it does
1. Picks 15 answers (5 each from draft scores 0, 1, 2), skipping rows whose reference was flagged.
2. Shows them in random order WITHOUT the draft score and WITHOUT telling you RAG / no RAG.
3. You type your own score. Quit any time (q); run again to continue.
4. At the end it shows how often you agree with the draft, lists the disagreements, and writes
   <file>_checked.csv with a `final_score` column (draft score, replaced by yours on checked rows).
"""
import argparse
import json
import random
import sys
import textwrap

import pandas as pd

import config

try:
    sys.stdout.reconfigure(errors="replace")
except Exception:
    pass

ap = argparse.ArgumentParser()
ap.add_argument("file", nargs="?", default=str(config.RESULTS_DIR / "generation_llama3.2_3b_draft_scored.csv"))
ap.add_argument("--n", type=int, default=15)
args = ap.parse_args()

df = pd.read_csv(args.file)
if "draft_score_claude" not in df.columns:
    sys.exit("This file has no draft_score_claude column. Use generation_llama3.2_3b_draft_scored.csv")
df["draft_note"] = df.get("draft_note", "").fillna("")

# ---- fixed, reproducible sample ----
ok = df[~df["draft_note"].str.startswith("REFERENCE")]
rng = random.Random(7)
sample = []
for s in (0, 1, 2):
    pool = list(ok[ok.draft_score_claude == s].index)
    rng.shuffle(pool)
    sample += pool[: args.n // 3]
rng.shuffle(sample)

progress_path = config.RESULTS_DIR / "spot_check.csv"
mine = {}
if progress_path.exists():
    prev = pd.read_csv(progress_path)
    mine = {int(r.row): int(r.my_score) for r in prev.itertuples()}


def save_progress():
    pd.DataFrame({"row": list(mine), "my_score": list(mine.values())}).to_csv(progress_path, index=False)


evidence = {}
if config.TESTSET_PATH.exists():
    evidence = {t["question"]: t.get("evidence", "") for t in
                json.loads(config.TESTSET_PATH.read_text(encoding="utf-8"))}


def wrap(text):
    return textwrap.fill(str(text), width=100, subsequent_indent="   ")


print(f"\nSpot-check of {len(sample)} answers. Already done: {len(mine)}\n")
print("RUBRIC:  2 = correct (same key fact as the reference)")
print("         1 = partly right (missing a key detail, or a wrong extra detail)")
print("         0 = wrong, irrelevant, or 'could not find'")
print("Judge against the REFERENCE. Ignore style and length.")
print("KEYS:    0 / 1 / 2 = score | b = back | s = skip | q = save and quit\n")

todo = [i for i in sample if i not in mine]
k = 0
while k < len(todo):
    i = todo[k]
    row = df.loc[i]
    print("=" * 100)
    print(f"[{len(mine) + 1}/{len(sample)}]")
    print("QUESTION : " + wrap(row["question"]))
    print("REFERENCE: " + wrap(row["reference"]))
    if evidence.get(row["question"]):
        print("EVIDENCE : " + wrap(evidence[row["question"]]))
    print("\nANSWER   : " + wrap(row["answer"]) + "\n")
    c = input("Your score [0/1/2, b, s, q]: ").strip().lower()
    if c in ("0", "1", "2"):
        mine[i] = int(c)
        save_progress()
        k += 1
    elif c == "b" and k > 0:
        k -= 1
        mine.pop(todo[k], None)
        save_progress()
    elif c == "s":
        k += 1
    elif c == "q":
        break
    else:
        print("Please type 0, 1, 2, b, s or q.")

done = [i for i in sample if i in mine]
if len(done) < len(sample):
    print(f"\nSaved. {len(done)} of {len(sample)} done. Run the same command again to continue.")
    sys.exit()

# ---- comparison ----
agree = [i for i in done if mine[i] == df.at[i, "draft_score_claude"]]
within1 = [i for i in done if abs(mine[i] - df.at[i, "draft_score_claude"]) <= 1]
print("\n" + "=" * 100)
print(f"Exact agreement with the draft : {len(agree)} of {len(done)} ({100 * len(agree) / len(done):.0f}%)")
print(f"Within one point               : {len(within1)} of {len(done)} ({100 * len(within1) / len(done):.0f}%)\n")
dis = [i for i in done if i not in agree]
if dis:
    print("DISAGREEMENTS (your score vs my draft):")
    for i in dis:
        print(f"  row {i}: you {mine[i]} | draft {df.at[i, 'draft_score_claude']} | {df.at[i, 'question'][:80]}")
        if df.at[i, "draft_note"]:
            print(f"           draft note: {df.at[i, 'draft_note']}")

df["final_score"] = df["draft_score_claude"]
df["spot_checked"] = False
for i, v in mine.items():
    df.at[i, "final_score"] = v
    df.at[i, "spot_checked"] = True
out = args.file.replace(".csv", "_checked.csv")
df.to_csv(out, index=False, encoding="utf-8")
print(f"\nSaved {out}")
if len(dis) > 3:
    print("You disagree on more than 3 of 15, so regrade everything yourself:  python grade_answers.py")
else:
    print("Good agreement. Now run:\n  python summarize_generation.py " + out + " --score-col final_score")
