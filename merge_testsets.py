"""Merge several test-set JSON files, drop duplicates, renumber ids.

Usage:
  python merge_testsets.py data/testset_v1.json data/testset.json --out data/testset_merged.json

Two questions count as duplicates if they share the same evidence passage or their
wording is very similar. The FIRST file's question wins, so list your reviewed file first.
Afterwards: review the merged file by hand, then rename it to data/testset.json.
"""
import argparse
import json
from difflib import SequenceMatcher
from pathlib import Path

from metrics import norm


def is_dup(a: dict, b: dict, threshold: float) -> bool:
    ea, eb = norm(a["evidence"]), norm(b["evidence"])
    if ea == eb or ea in eb or eb in ea:
        return True
    return SequenceMatcher(None, norm(a["question"]), norm(b["question"])).ratio() >= threshold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", default="data/testset_merged.json")
    ap.add_argument("--threshold", type=float, default=0.85, help="question similarity for duplicates")
    args = ap.parse_args()

    merged, dropped = [], []
    for f in args.files:
        for item in json.loads(Path(f).read_text(encoding="utf-8")):
            match = next((m for m in merged if is_dup(m, item, args.threshold)), None)
            if match:
                dropped.append((item, match, f))
            else:
                merged.append(item)

    for i, m in enumerate(merged, 1):
        m["id"] = i
    Path(args.out).write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Kept {len(merged)} questions, dropped {len(dropped)} duplicates -> {args.out}\n")
    for item, match, f in dropped:
        print(f"DROPPED ({Path(f).name}): {item['question']}\n   duplicate of: {match['question']}\n")
    per_act = {}
    for m in merged:
        per_act[m["source"]] = per_act.get(m["source"], 0) + 1
    print("Questions per file:", per_act)


if __name__ == "__main__":
    main()
