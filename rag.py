"""End-to-end RAG pipeline (retrieve -> build prompt -> generate) with timing."""
import time

from llm import LLM
from retriever import Index

NOT_FOUND = "I could not find this in the provided documents."

RAG_SYSTEM = (
    "You are a legal information assistant for Indian law. Answer the question using ONLY "
    "the numbered context passages. Cite the passages you used like [1] or [2]. "
    f"If the answer is not in the context, reply exactly: \"{NOT_FOUND}\" "
    "Be concise, answer in at most two sentences, and do not add facts that are not in the context."
)

NO_RAG_SYSTEM = (
    "You are a legal information assistant for Indian law. Answer in at most two sentences."
)


def build_context(hits: list[dict]) -> str:
    return "\n\n".join(
        f"[{i + 1}] (Source: {h['source']}, page {h['page']})\n{h['text']}"
        for i, h in enumerate(hits)
    )


class RAGPipeline:
    def __init__(self, index: Index, llm: LLM):
        self.index = index
        self.llm = llm

    def answer(self, question: str, k: int = 5, mode: str = "hybrid",
               rerank: bool = False, use_rag: bool = True, max_tokens: int = 400) -> dict:
        t0 = time.perf_counter()
        hits = self.index.retrieve(question, k=k, mode=mode, rerank=rerank) if use_rag else []
        t1 = time.perf_counter()

        if use_rag:
            prompt = f"Context:\n{build_context(hits)}\n\nQuestion: {question}\n\nAnswer:"
            system = RAG_SYSTEM
        else:
            prompt, system = f"Question: {question}\n\nAnswer:", NO_RAG_SYSTEM

        answer = self.llm.generate(prompt, system=system, max_tokens=max_tokens)
        t2 = time.perf_counter()
        return {
            "question": question,
            "answer": answer,
            "hits": hits,
            "contexts": [h["text"] for h in hits],
            "retrieval_s": t1 - t0,
            "generation_s": t2 - t1,
            "total_s": t2 - t0,
        }
