"""Streamlit demo UI.   Run:  streamlit run app.py"""
import streamlit as st

import config
from ingest import chunk_documents, load_documents
from llm import LLM
from rag import RAGPipeline
from retriever import Index

st.set_page_config(page_title="Indian Legal RAG", page_icon="⚖️", layout="wide")
st.title("Indian Legal QA — Retrieval-Augmented Generation")
st.caption("Answers are generated only from the indexed documents and cite their sources. "
           "This is a research prototype, not legal advice.")


@st.cache_resource(show_spinner="Reading documents and building the index...")
def load_index(chunk_size: int, overlap: int, embedder: str) -> Index:
    chunks = chunk_documents(load_documents(config.RAW_DIR), chunk_size, overlap)
    return Index(chunks, config.EMBEDDERS[embedder], show_progress=False)


with st.sidebar:
    st.header("Settings")
    backend = st.selectbox("LLM backend", ["ollama", "anthropic", "gemini", "hf"],
                           index=["ollama", "anthropic", "gemini", "hf"].index(config.LLM_BACKEND))
    embedder = st.selectbox("Embedding model", list(config.EMBEDDERS), index=1)
    chunk_size = st.select_slider("Chunk size (words)", [100, 200, 400], value=config.DEFAULT_CHUNK_SIZE)
    overlap = max(chunk_size // 5, 10)
    mode = st.selectbox("Retrieval", ["hybrid", "dense", "bm25"])
    rerank = st.checkbox("Cross-encoder reranking", value=True)
    k = st.slider("Top-k passages", 1, 10, config.DEFAULT_K)
    compare = st.checkbox("Also show answer WITHOUT retrieval (baseline)")

try:
    index = load_index(chunk_size, overlap, embedder)
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

pipe = RAGPipeline(index, LLM(backend))
st.sidebar.write(f"Indexed chunks: **{len(index.chunks)}**")

question = st.text_input("Ask a question about the indexed Acts",
                         placeholder="e.g. What remedies are available to a consumer under the Consumer Protection Act, 2019?")

if question:
    try:
        with st.spinner("Retrieving and generating..."):
            res = pipe.answer(question, k=k, mode=mode, rerank=rerank)
            base = pipe.answer(question, use_rag=False) if compare else None
    except Exception as e:
        st.error(f"LLM call failed: {e}")
        st.stop()

    cols = st.columns(2) if base else [st.container()]
    with cols[0]:
        st.subheader("RAG answer")
        st.write(res["answer"])
        c1, c2, c3 = st.columns(3)
        c1.metric("Retrieval", f"{res['retrieval_s'] * 1000:.0f} ms")
        c2.metric("Generation", f"{res['generation_s']:.1f} s")
        c3.metric("Total", f"{res['total_s']:.1f} s")
    if base:
        with cols[1]:
            st.subheader("LLM only (no retrieval)")
            st.write(base["answer"])
            st.caption("No sources: cannot be verified, may be hallucinated.")

    st.subheader("Retrieved sources")
    for i, h in enumerate(res["hits"], 1):
        with st.expander(f"[{i}] {h['source']} — page {h['page']}  (score {h['score']:.3f})"):
            st.write(h["text"])
