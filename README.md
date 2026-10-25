# RAG Legal Question Answering System

## Project Overview

This project explores the implementation and performance of a Retrieval-Augmented Generation (RAG) system for domain-specific question answering using Indian legal documents.

The system retrieves relevant passages from a collection of legal documents and provides them as context to a language model to generate answers supported by source references.

## Legal Documents

The project focuses on three Indian Acts:

- Consumer Protection Act, 2019
- Information Technology Act, 2000
- Right to Information Act, 2005

## Technologies

- Python
- Retrieval-Augmented Generation (RAG)
- Sentence embeddings
- BM25 keyword retrieval
- Hybrid retrieval with Reciprocal Rank Fusion (RRF)
- Cross-encoder reranking
- Llama 3.2 3B Instruct through Ollama
- Streamlit

## Main Features

- Document ingestion and chunking
- Dense and keyword-based retrieval
- Hybrid retrieval
- Optional reranking of retrieved passages
- Context-based answer generation
- Source references in generated answers
- Retrieval and answer-quality evaluation

## Project Structure

The source code is organized into Python modules for document ingestion, retrieval, language-model interaction, RAG orchestration, evaluation, and the Streamlit interface.

Refer to the actual files in this repository for the implementation.

## Requirements

Install the dependencies listed in `requirements.txt`. The project also requires the appropriate Ollama installation and language model if you want to run local answer generation.

## Limitations

This is an academic prototype. Answer quality depends on retrieving relevant evidence, and generated answers may still contain errors. The system should not replace professional legal advice.

## Academic Project

**Project Title:** Implementation and Performance Analysis of Retrieval-Augmented Generation (RAG) Architecture for Domain-Specific Question Answering Systems

**Program:** Bachelor of Computer Applications (Honours/Research)

## Disclaimer

This project is intended for academic and educational purposes. Verify legal information against official sources before relying on it.
