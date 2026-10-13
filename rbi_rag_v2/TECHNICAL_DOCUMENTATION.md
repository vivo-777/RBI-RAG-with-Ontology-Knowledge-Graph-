# RBI Regulatory Assistant: Comprehensive Architecture Deep Dive

This document provides a ground-up explanation of the Retrieval-Augmented Generation (RAG) pipeline built for the RBI Regulatory Assistant. It is designed to give executives, compliance officers, and engineering teams a complete understanding of how the system ingests data, retrieves context, and verifies factual correctness.

*Last updated: 2026-07-09 — Major architecture overhaul incorporating Groq routing, DSPy pipelines, dual-check verification, and structured table extraction.*

---

## 1. System Overview

The **RBI Regulatory Assistant** is a production-grade AI search and chat application specifically tuned for interpreting Reserve Bank of India (RBI) circulars. 

Instead of relying on a black-box AI, this system uses a **Retrieval-Augmented Generation (RAG)** architecture. It finds the exact paragraphs in the RBI circulars and passes *only* those to the AI. It features a citation-verified architecture with measured faithfulness (see `test_results.md` for evaluation metrics).

### Tech Stack & Deployment Variants
The system supports two deployment variants:
* **Cloud-API Variant (Default)**: Uses Groq (`llama-3.3-70b-versatile`) for primary Generation and Groq (`llama-3.1-8b-instant`) for ultra-low latency routing, evaluation metrics, and factual verification.
* **On-Premise / Tailscale Variant**: Uses locally hosted `qwen2.5:7b-instruct` via Ollama for all generation tasks. To leverage remote GPU clusters while maintaining local data processing, LLM calls are securely routed through an **SSH Tunnel** (e.g., `ssh -N -L 11434:localhost:11434 user@100.x.y.z`) to a Tailscale node.
* **Common Components**:
  * **Frontend:** Next.js, React, Tailwind CSS (running on port 3001)
  * **Backend:** Python, FastAPI (running on port 8543)
  * **Embedding Model:** `BAAI/bge-small-en-v1.5` (CUDA optimized)
  * **Reranking Model:** `BAAI/bge-reranker-v2-m3` (CUDA optimized)
  * **Evaluation Framework:** DSPy & RAGAS for rigorous pipeline scoring.

---

## 2. The Hybrid Database Architecture

The system uses three specialized search engines:

### A. Qdrant (The Dense Vector Database)
* **What it is:** A vector database for semantic meaning.
* **How it works:** Paragraphs are embedded into 384-dimensional vectors. Matches concepts conceptually (e.g., "minimum capital" matches "CRAR 9%").
* **Where it lives:** Stored locally in `qdrant_data/`.

### B. BM25 (The Sparse Keyword Database)
* **What it is:** A keyword-matching engine.
* **How it works:** Excels at exact acronyms (e.g., "SMA-1").
* **Where it lives:** Serialized file `bm25_data.pkl`.

### C. SQLite (Facts DB & Ontology)
* **What it is:** A relational database (`facts.db`) and in-memory graph (`graph_store.py`).
* **How it works:** 
  1. Stores structured tabular facts extracted from circulars (entity, metric, value, date) for precise verification.
  2. Maps relationships like supersession and domain metadata.

---

## 3. The Data Ingestion Pipeline

1. **PDF Parsing:** Uses `pdfplumber` to extract text and tables.
2. **Structure-Aware Chunking:** Breaks circulars into large, robust chunks (400-word limit, 150-word overlap) to ensure critical clauses and multi-sentence context are kept together without getting severed.
3. **Structured Table Extraction (Optional):** Tables are optionally passed to Groq/Ollama to extract JSON arrays of numeric facts, which are inserted directly into the SQLite `structured_facts` table. This step can be disabled during fast re-ingestion to eliminate latency bottlenecks.
4. **Embedding & Indexing:** Chunks are saved into Qdrant and BM25. The circular is registered in the Ontology Graph.

---

## 4. The Query & Retrieval Pipeline

1. **Pre-Routing Classification:** An LLM evaluates if the query is Simple, Compound, or Out-Of-Context. 
2. **Decomposition & Parallel Retrieval:** Compound queries are decomposed. For each sub-query (or the main query), the pipeline fetches deep candidate pools (Top 40 Dense + Top 40 Sparse) using Qdrant and BM25.
3. **Retrieval-Gated Classification Override:** Even if the classifier flags a query as `OUT_OF_CONTEXT` (which often happens erroneously with niche acronyms like DAY-NRLM), the system executes retrieval anyway. If the reranker surfaces high-confidence chunks (score ≥ 0.5), the `OUT_OF_CONTEXT` label is overridden and generation proceeds.
4. **Ontology Boost & Supersession:** Retrieved circulars get a 20% score boost if their metadata matches the query domain. Superseded circulars are dynamically flagged with a `[WARNING: SUPERSEDED BY...]` badge.
5. **Reciprocal Rank Fusion (RRF):** Dense and sparse results are mathematically merged.
6. **Cross-Encoder Reranking:** The unioned results are reranked using `bge-reranker-v2-m3` (Top 20). 
7. **Chunk Quality Filter:** Chunks containing fewer than 15 words (typically Table of Contents entries, generic headers, or stray metadata) are stripped out post-reranking. This forces the LLM to only rely on substantive paragraphs.
8. **Abstention Gate:** If no substantive results are found, the system aborts generation to prevent hallucination.
9. **Generation (DSPy):** The synthesized prompt is sent to the LLM (e.g. `qwen2.5:7b-instruct` via SSH tunnel or Groq). Three separate answer generation streams (temperature 0.7) can be fired for self-consistency voting.

---

## 5. Security, Verification & Audit Trails

* **Dual-Check Verifier**: Before returning an answer, the pipeline analyzes every claim:
  * **Check A (Facts DB)**: Numeric claims are verified against the SQLite structured facts store.
  * **Check B (Text Grounding)**: Claims are verified via cosine similarity (Jaccard + Dense) against the source chunk text.
* **Adversarial Self-Critique**: Numeric/high-risk claims are critiqued by Groq. If Groq finds a logical flaw, the claim is demoted.
* **Self-Consistency Voting**: If the three parallel generations disagree on a claim, it is tagged `[⚠️ Requires Analyst Review]`.
* **Audit Trail Logging**: Every query writes an immutable JSON log to `audit_logs/` capturing the query, retrieved IDs, model version, exact prompt, claim verifier labels, and latency.

## 5. DSPy & Groq Optimization Strategies

To handle Groq's extremely strict Tokens-Per-Minute (TPM) and Tokens-Per-Day (TPD) limits on the free tier, the system employs the following optimizations:
* **Throttled Execution:** An artificial 35-second pause is injected between evaluation queries to ensure the rolling 60-second window does not exceed 12,000 TPM.
* **Context Pruning:** Retaining only the top 5 reranked chunks (down from 8) drastically reduces prompt size. A dynamic token budget (`MAX_CONTEXT_TOKENS = 9,400`) enforces a hard ceiling — chunks are added in relevance order until the budget is exhausted, with the last chunk truncated if needed. This prevents 'Request Too Large' errors on the 70B model while ensuring the highest-relevance chunks are always included.
* **Output Token Cap:** Max output tokens reduced from 4,096 → 2,048 for the 70B model (regulatory answers are typically 200-500 tokens). This frees ~2K tokens of TPM budget for richer context.
* **Model Routing for Evaluation:** To bypass the 100,000 TPD limit on llama-3.3-70b-versatile, the evaluation framework strictly routes factual verification (Faithfulness/Precision metrics) to the llama-3.1-8b-instant model. This reduces token consumption per query from ~55,000 tokens down to ~5,000 tokens on the 70B model, preserving daily quota while maintaining high precision for the actual answer generation.
* **Disabled Self-Consistency:** DSPy's default multi-vote generation (temperature=0.7) is disabled to prevent 3x token multiplication per query.

---

## 6. Recent Architectural Enhancements (July 2026)

To specifically address edge-case failures, false abstentions, and strict domain nuances, the architecture was recently upgraded:

* **FinBERT Relevance Fusion**: Generic embedding models often fail to discern critical financial nuances. The retrieval pipeline now passes Qdrant chunks through `ProsusAI/finbert` to generate a financial-domain semantic relevance score. This score is mathematically fused (at 30% weight) with the standard cross-encoder during Reciprocal Rank Fusion, ensuring that deep regulatory chunks surface to the top.
* **Pre-Retrieval Acronym Expansion**: A static acronym map expands notoriously problematic abbreviations (e.g., `DAY-NRLM` → `Deendayal Antyodaya Yojana - National Rural Livelihoods Mission`, `KCC`, `PMJDY`) before they hit the dense index.
* **Relaxed Validation & Partial Matching**: The `COSINE_SIMILARITY_THRESHOLD` in the `citation_verifier` was lowered from 0.70 to 0.55. A secondary fallback checks for direct keyword overlap if similarity lands in the grey zone, dramatically reducing false abstentions for valid answers.
* **Smart Sentence Chunking**: Chunking algorithms were adjusted to look behind for strict sentence boundaries (periods) and the `overlap` was increased to 75 to ensure critical multi-sentence clauses are not severed.
* **Expanded Evaluation Suite**: The evaluation dataset was scaled from 21 to 40 queries, encompassing a mix of complex, compound, and tough edge-cases across 22 circulars. The system automatically evaluates these against the pipeline and generates an aggregate Word (`.docx`) report for comparative human review.

## 7. Deep Edge-Case Fixes (July 2026 - Update 2)

Following a diagnostic run of the 40-query dataset over an SSH tunnel, three critical root causes were identified and resolved to boost the success rate:

* **Retrieval-Gated Classifier Override**: LLMs (like Qwen 2.5) consistently mislabeled niche acronyms like "DAY-NRLM" as `OUT_OF_CONTEXT` because they lacked pre-training exposure to Indian financial schemes. The pipeline now *always* executes retrieval; if the cross-encoder reranker finds chunks scoring ≥0.5, the `OUT_OF_CONTEXT` label is overridden.
* **Deep Retrieval Pool + Chunk Quality Filtering**: When queries hit the exact right document, the retriever would often pull "shallow" chunks (like Table of Contents entries or generic section headers) that lacked the actual detailed answer. By increasing the retrieval pool depth (`TOP_K_DENSE` = 40) and adding a strict post-reranking filter that drops chunks with fewer than 15 words, the pipeline guarantees the LLM only sees substantive text.
* **Aggressive Re-Chunking**: The sliding window chunk size was doubled (from 200 to 400 words) with a larger overlap (150 words). This prevents dense regulatory formulas (like the ANBC calculation) from being severed across vector boundaries, ensuring the LLM has complete contiguous context.
