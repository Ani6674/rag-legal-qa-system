"""Retrieval metrics (evidence-based), generation metrics (ROUGE-L, BERTScore, LLM-judge)."""
import re


# ---------------- retrieval ----------------
def norm(text: str) -> str:
    return " ".join(re.findall(r"\w+", text.lower()))


def as_list(evidence) -> list[str]:
    return [evidence] if isinstance(evidence, str) else list(evidence)


def retrieval_scores(chunk_texts: list[str], evidence, k: int) -> dict:
    """A chunk is relevant if it contains a gold evidence span (chunk-size independent).

    hit@k    : 1 if any evidence span appears in the top-k chunks
    recall@k : fraction of evidence spans covered by the top-k chunks
    rr@k     : 1/rank of the first relevant chunk within top-k, else 0
    """
    evs = [norm(e) for e in as_list(evidence)]
    top = [norm(t) for t in chunk_texts[:k]]
    covered = {e for e in evs if any(e in t for t in top)}
    rr = 0.0
    for rank, t in enumerate(top, start=1):
        if any(e in t for e in evs):
            rr = 1.0 / rank
            break
    return {"hit": float(bool(covered)), "recall": len(covered) / len(evs), "rr": rr}


# ---------------- generation ----------------
def rouge_l(pred: str, ref: str) -> float:
    from rouge_score import rouge_scorer
    return rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True).score(ref, pred)["rougeL"].fmeasure


def bert_score_f1(preds: list[str], refs: list[str]) -> list[float]:
    from bert_score import score
    _, _, f1 = score(preds, refs, lang="en", verbose=False)
    return f1.tolist()


FAITHFULNESS_PROMPT = """You are grading a question-answering system.

Context:
{context}

Answer to grade:
{answer}

Is every claim in the answer supported by the context above?
5 = fully supported, 3 = partly supported, 1 = mostly unsupported or contradicts the context.
Reply with a single integer from 1 to 5 and nothing else."""

CORRECTNESS_PROMPT = """You are grading a question-answering system.

Question: {question}
Reference answer: {reference}
Candidate answer: {answer}

Does the candidate answer convey the same facts as the reference answer?
5 = same facts, 3 = partly correct or incomplete, 1 = wrong or irrelevant.
Reply with a single integer from 1 to 5 and nothing else."""


def judge(llm, prompt: str):
    """LLM-as-judge. Returns an int 1-5 or None if unparseable."""
    out = llm.generate(prompt, max_tokens=8, temperature=0.0)
    m = re.search(r"[1-5]", out)
    return int(m.group()) if m else None
