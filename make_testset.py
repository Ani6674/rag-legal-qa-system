"""Draft a test set (question, reference answer, evidence span) from your documents using an LLM.

- Balanced sampling across all Acts; the Act's name is given to the model.
- Vague questions are rejected automatically.
- Live progress line with elapsed time and ETA.
- Saves after EVERY accepted question: press Ctrl+C any time and nothing is lost.

IMPORTANT: still review data/testset.json by hand and write ~15-20 questions yourself.
LLM-drafted questions copy wording from the passage, which favours lexical (BM25)
retrieval - discuss this bias in your report.

Usage:  python make_testset.py --n 30 --llm ollama
"""
import argparse
import json
import random
import re
import sys
import time

import config
from ingest import chunk_documents, load_documents
from llm import LLM
from metrics import norm

PROMPT = """Read this passage from the {act}, an Indian law.

Passage:
{passage}

Write ONE question that can be answered fully from this passage alone, a concise reference answer, and an evidence span.
Rules:
- The question MUST mention "{act_short}" by name and be understandable on its own.
- Never say "the passage", "this section" or "the text".
- Ask about a specific rule, definition, time limit, penalty, power or duty (not "what is the name of the Act").
- "evidence" must be copied EXACTLY, word for word, as 8 to 20 consecutive words from the passage that support the answer.
Return only JSON: {{"question": "...", "answer": "...", "evidence": "..."}}"""

BAD_PHRASES = ("the passage", "this section", "the text", "this act?", "name of the act")


def fmt(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"


def good_chunk(text: str) -> bool:
    """Skip tables of contents, tiny passages and number-heavy boilerplate."""
    if len(text.split()) < 120 or "....." in text:
        return False
    return sum(ch.isalpha() for ch in text) / len(text) > 0.75


def balanced(chunks, seed):
    """Shuffle chunks within each source, then interleave sources round-robin."""
    rng = random.Random(seed)
    groups = {}
    for c in chunks:
        groups.setdefault(c["source"], []).append(c)
    for g in groups.values():
        rng.shuffle(g)
    out, lists = [], list(groups.values())
    while any(lists):
        for g in lists:
            if g:
                out.append(g.pop())
    return out


def question_ok(q: str, aliases: list[str]) -> bool:
    ql = q.lower()
    if len(q.split()) < 7:
        return False
    if any(b in ql for b in BAD_PHRASES):
        return False
    return not aliases or any(a in ql for a in aliases)


def save(tests):
    config.TESTSET_PATH.write_text(json.dumps(tests, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--llm", default=None, help="ollama | anthropic | gemini | hf")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    docs = load_documents(config.RAW_DIR)
    chunks = [c for c in chunk_documents(docs, 200, 0) if good_chunk(c["text"])]
    chunks = balanced(chunks, args.seed)
    llm = LLM(args.llm)
    print(f"{len(chunks)} usable passages | LLM: {llm.name} | target: {args.n} questions")
    print("Press Ctrl+C any time: questions found so far are already saved.\n")

    tests, tried, max_tries = [], 0, args.n * 8
    start = time.perf_counter()
    try:
        for c in chunks:
            if len(tests) >= args.n or tried >= max_tries:
                break
            tried += 1
            act, aliases = config.ACT_NAMES.get(c["source"], (c["source"], []))
            act_short = act.split(",")[0]
            out = llm.generate(PROMPT.format(act=act, act_short=act_short, passage=c["text"]),
                               max_tokens=250, json_mode=True)

            item = None
            m = re.search(r"\{.*\}", out, re.DOTALL)
            if m:
                try:
                    d = json.loads(m.group())
                    q, a, ev = d["question"], d["answer"], d["evidence"]
                    if (isinstance(q, str) and isinstance(a, str) and isinstance(ev, str)
                            and question_ok(q, aliases) and norm(ev) in norm(c["text"])):
                        item = {"id": len(tests) + 1, "question": q, "reference_answer": a,
                                "evidence": ev, "source": c["source"], "page": c["page"],
                                "origin": "llm_drafted", "reviewed": False}
                except (json.JSONDecodeError, KeyError, TypeError):
                    pass

            if item:
                tests.append(item)
                save(tests)
                print("\r" + " " * 90 + f"\r[{len(tests)}/{args.n}] ({act_short}) {item['question']}")

            # live progress + ETA (smoothed acceptance rate so it works from the first attempt)
            elapsed = time.perf_counter() - start
            per_try = elapsed / tried
            rate = (len(tests) + 1) / (tried + 2)
            eta = per_try * (args.n - len(tests)) / rate
            sys.stdout.write(f"\r  tried {tried} | accepted {len(tests)}/{args.n} | "
                             f"{per_try:.1f}s per try | elapsed {fmt(elapsed)} | ETA ~{fmt(eta)}   ")
            sys.stdout.flush()
    except KeyboardInterrupt:
        print("\n\nStopped by you (Ctrl+C).")

    save(tests)
    print(f"\n\nSaved {len(tests)} questions ({tried} passages tried, {fmt(time.perf_counter() - start)}) "
          f"to {config.TESTSET_PATH}")
    print("Now review them by hand and set \"reviewed\": true.")


if __name__ == "__main__":
    main()
