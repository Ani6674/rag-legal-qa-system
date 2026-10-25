# Implementation and Performance Analysis of RAG for Domain-Specific QA (Indian Legal Domain)

Author: Aniket Kumar | Enrollment: A35404824056 | Area: Generative AI & LLMs

## Pipeline
PDF/TXT -> clean -> chunk (sliding window) -> embeddings (MiniLM / BGE-small) -> FAISS + BM25
-> dense | BM25 | hybrid (RRF) -> optional cross-encoder rerank -> prompt -> LLM -> answer with [n] citations

## Files
| File | Purpose |
|---|---|
| `ingest.py` | load PDFs, clean, chunk |
| `retriever.py` | FAISS dense, BM25, hybrid (RRF), reranker |
| `llm.py` | backends: ollama, anthropic, gemini, hf |
| `rag.py` | RAG pipeline + no-RAG baseline, timing |
| `metrics.py` | Hit@k, Recall@k, MRR, ROUGE-L, BERTScore, LLM-judge |
| `make_testset.py` | draft test set from your documents (then review by hand) |
| `run_retrieval_experiments.py` | grid: chunk size x embedder x mode x rerank x k |
| `run_generation_experiments.py` | no-RAG vs RAG variants, per LLM |
| `plots.py` | figures for the report |
| `app.py` | Streamlit demo |

## Setup
```bash
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Python version
Works on Python 3.10+. If `pip install` fails on a package for 3.15, install the rest and
retry that package separately (`faiss-cpu` is optional; `bert-score` can be skipped with `--no-bertscore`).

## 1. Data
Download 3-5 Acts as PDFs from https://www.indiacode.nic.in (e.g. Consumer Protection Act 2019,
Information Technology Act 2000, Right to Information Act 2005, Constitution of India)
and put them in `data/raw/`. Keep total size modest (a few hundred pages) so indexing is fast.

## 2. LLM
Pick ONE backend (set env vars; see `config.py`):
- Local free: install Ollama, `ollama pull llama3.2:3b`, then `export LLM_BACKEND=ollama`
- Anthropic API: `export LLM_BACKEND=anthropic ANTHROPIC_API_KEY=...`
- Gemini API: `export LLM_BACKEND=gemini GEMINI_API_KEY=...` (check the current model name)
- Colab GPU: `export LLM_BACKEND=hf` (downloads Qwen2.5-1.5B-Instruct)

## 3. Test set (most important for credibility)
```bash
python make_testset.py --n 50 --llm anthropic
```
Then open `data/testset.json`, fix wrong questions/answers, and ADD 15-20 questions you write yourself
(see `data/testset_format_example.json`). Set `"reviewed": true`.
`evidence` must be an exact span from the document; it is what decides whether a retrieved chunk is relevant.

## 4. Experiments
```bash
python run_retrieval_experiments.py                                   # no LLM needed
python run_generation_experiments.py --llm ollama --judge-llm anthropic
python run_generation_experiments.py --llm anthropic --judge-llm anthropic   # second LLM to compare
python plots.py
```
Results: `results/*.csv`, figures: `results/*.png`.

## 5. Demo
```bash
streamlit run app.py
```

## Known limitations (put these in your report)
- Chunk size is in words, not tokens. MiniLM truncates at ~256 word-pieces, so large chunks lose their tail.
- Evidence-span matching can miss a hit if the span straddles a chunk boundary (overlap reduces this).
- LLM-drafted questions share wording with passages, which favours BM25; hence the manual questions.
- LLM-as-judge is noisy; use a stronger judge than the evaluated model and report it.
- Small test set: report results as indicative, not statistically conclusive.
