# RBI Regulatory Assistant

A production-grade Retrieval-Augmented Generation (RAG) pipeline designed for precise, factual, and citation-verified search across Reserve Bank of India (RBI) regulatory circulars.

## Overview

Unlike generic AI chat interfaces, the RBI Regulatory Assistant is built with a rigorous **Retrieval-Augmented Generation (RAG)** architecture. It ingests complex regulatory PDFs, extracts semantic concepts and structured tables, and uses a multi-stage retrieval pipeline to ground every LLM response in exact, verified clauses from official documents.

It is designed to give executives, compliance officers, and engineering teams a complete, hallucination-free understanding of regulatory texts.

## Key Features

* **Hybrid Search Engine:** Combines **Qdrant** (Dense Vector Search via `BAAI/bge-small-en-v1.5`) for semantic conceptual matching with **BM25** (Sparse Search) for exact acronyms and keywords.
* **Graph Ontology & Structured Facts:** Extracted facts (entities, metrics, dates) are stored in SQLite, and relationships (like circular supersessions) are mapped in an in-memory graph.
* **FinBERT Relevance Fusion:** Embeddings are augmented with `ProsusAI/finbert` to generate financial-domain semantic relevance scores, which are mathematically fused with standard cross-encoders (`BAAI/bge-reranker-v2-m3`).
* **Dual-Check Fact Verification:** Every response undergoes a strict verification phase against structured DB facts and textual grounding (Cosine Similarity + Jaccard Index) before surfacing to the user.
* **Flexible LLM Deployment:**
  * **Cloud Variant:** Uses Groq (`llama-3.3-70b-versatile`) for ultra-fast, high-quality generation.
  * **On-Premise Variant:** Routes generation to local models like `qwen2.5:7b-instruct` via Ollama and Tailscale SSH tunnels.

## Repository Structure

The core source code lives inside the `rbi_rag_v2` directory.

* `rbi_rag_v2/api/` - FastAPI backend servers for handling search and chat queries.
* `rbi_rag_v2/eval/` - Rigorous DSPy and RAGAS evaluation pipelines and ground-truth datasets.
* `rbi_rag_v2/generation/` - LLM client integrations (Groq, Ollama), prompt synthesizers, and fact verifiers.
* `rbi_rag_v2/ingestion/` - PDF parsing (`pdfplumber`), structure-aware chunking, and database indexing.
* `rbi_rag_v2/ontology/` - SQLite facts store and knowledge graph builder for tracking supersessions and domain metadata.
* `rbi_rag_v2/retrieval/` - Query decomposition, hybrid search execution, and reciprocal rank fusion.
* `rbi_rag_v2/web/` - Next.js/React frontend application for interacting with the assistant.
* `rbi_rag_v2/scripts/` - Utility scripts for one-off evaluations, debugging, document conversion, and database purging.

## Documentation

For a deep dive into the mathematical definitions and architectural workflows, please review the internal documentation:
* [Technical Documentation](rbi_rag_v2/TECHNICAL_DOCUMENTATION.md)
* [Pipeline Explainer (Math & Mermaid Graphs)](rbi_rag_v2/PIPELINE_EXPLAINER.md)

## Tech Stack

* **Backend:** Python 3, FastAPI, Pydantic
* **Frontend:** Next.js, React, Tailwind CSS
* **Databases:** Qdrant (Vectors), SQLite (Facts/Ontology), Pickle (BM25)
* **AI/ML:** HuggingFace Transformers (Embeddings/Reranking), Groq, Ollama, DSPy
