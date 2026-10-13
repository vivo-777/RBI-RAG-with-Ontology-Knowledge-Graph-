"""
Evaluation harness for the DSPy RBI Advisor.
Runs the GroundedRBIAdvisor module against queries.json and outputs metrics.
"""

import json
import os
import sys
import time
import logging
import numpy as np

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from generation.dspy_advisor import generate_dspy_answer
from generation.citation_verifier import is_non_coverage_sentence, verify_sentence

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_queries():
    queries_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "queries.json")
    if os.path.exists(queries_path):
        with open(queries_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info("Loaded %d queries from %s", len(data), queries_path)
        return data
    else:
        logger.error("queries.json not found!")
        return []

def calculate_metrics(answer, context_chunks, expected_citations, retrieved_files):
    # 1. Faithfulness
    import re
    sentences = [s.strip() for s in re.split(r'(?<=[.!?]) +', answer) if s.strip()]
    substantive_sentences = [s for s in sentences if not is_non_coverage_sentence(s)]
    
    if not substantive_sentences:
        faithfulness = 1.0
    else:
        entailed_count = 0
        for sent in substantive_sentences:
            result = verify_sentence(sent, context_chunks)
            if result in ("supported", "partially_supported"):
                entailed_count += 1
        faithfulness = entailed_count / len(substantive_sentences)
        
    # 2. Context Recall
    if not expected_citations:
        context_recall = 1.0
    else:
        retrieved_set = set(retrieved_files)
        matched = sum(1 for expected in expected_citations if expected in retrieved_set)
        context_recall = matched / len(expected_citations)
        
    # 3. Context Precision
    if not expected_citations:
        context_precision = 1.0
    else:
        precision_scores = []
        hits = 0
        for idx, filename in enumerate(retrieved_files, start=1):
            if filename in expected_citations:
                hits += 1
                precision_scores.append(hits / idx)
        context_precision = np.mean(precision_scores) if precision_scores else 0.0
        
    return faithfulness, context_recall, context_precision

def run_eval():
    # Clear DSPy cache on disk before testing
    import shutil
    dspy_cache_dir = os.path.expanduser("~/.dspy_cache")
    if os.path.exists(dspy_cache_dir):
        try:
            shutil.rmtree(dspy_cache_dir)
            print(f"Deleted DSPy pre-fetched cache directory: {dspy_cache_dir}")
        except Exception as e:
            print(f"Could not delete cache directory: {e}")

    queries = load_queries()
    if not queries:
        return
        
    total = len(queries)
    
    state_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_state.json")
    
    all_faithfulness = []
    all_recall = []
    all_precision = []
    all_latency = []
    query_details = []
    start_index = 0
    
    # Load state if exists
    if os.path.exists(state_file):
        try:
            with open(state_file, "r") as f:
                state = json.load(f)
                all_faithfulness = state.get("all_faithfulness", [])
                all_recall = state.get("all_recall", [])
                all_precision = state.get("all_precision", [])
                all_latency = state.get("all_latency", [])
                query_details = state.get("query_details", [])
                start_index = len(query_details)
                print(f"Resuming evaluation from query {start_index + 1}...")
        except Exception as e:
            print(f"Could not load state: {e}")
    
    for i in range(start_index, total):
        entry = queries[i]
        question = entry.get("question", "")
        expected_citations = entry.get("expected_citations", [])
        q_type = entry.get("type", "factual")
        
        print(f"Evaluating {i + 1}/{total}: {question[:60]}...", flush=True)
        
        # Call the unified pipeline
        res = generate_dspy_answer(question)
        answer = res.get("answer", "")
        latency = res.get("latency", 0)
        retrieved_chunks = res.get("retrieved_chunks", [])
        
        # Detect catastrophic failure (API limits)
        if answer.startswith("Error:") or not answer.strip():
            print("\n=======================================================")
            print(f"API EXHAUSTED OR FATAL ERROR ON QUERY {i + 1}!")
            print("The script has securely saved your progress.")
            print(">>> PAUSED: Update your Groq API key in .env, then press ENTER to continue <<<")
            print("=======================================================")
            input()  # Block until user presses Enter
            
            # Reload .env to pick up the new key
            from dotenv import load_dotenv
            load_dotenv(dotenv_path=os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"
            ), override=True)
            config.GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
            
            # Re-initialize the Groq client with the new key
            from generation import groq_client
            from groq import Groq
            groq_client.client = Groq(api_key=config.GROQ_API_KEY)
            print(f"Reloaded API key. Retrying query {i + 1}...")
            
            # Retry the same query (don't increment i)
            i -= 1
            continue
        
        # Throttling for Groq rate limits (~14400 TPM for llama-3.3-70b-versatile)
        # We MUST have this 35-second gap, otherwise Groq's Free Tier blocks the request and fails the test.
        time.sleep(35)
        
        retrieved_files = [c["payload"].get("filename", "") for c in retrieved_chunks]
        context_chunks = [c["payload"].get("content", "") for c in retrieved_chunks]
        
        # Calculate metrics
        faithfulness, context_recall, context_precision = calculate_metrics(
            answer, context_chunks, expected_citations, retrieved_files
        )
        
        all_faithfulness.append(faithfulness)
        all_recall.append(context_recall)
        all_precision.append(context_precision)
        all_latency.append(latency)
        
        # Sources formatting (limit to top 3)
        sources_list = []
        seen_files = set()
        top_docs = []
        for chunk in retrieved_chunks:
            payload = chunk["payload"]
            filename = payload.get("filename", "Unknown")
            page = payload.get("page_number", 1)
            
            if filename not in seen_files:
                seen_files.add(filename)
                top_docs.append((filename, page))
            
            if len(top_docs) == 3:
                break
                
        for idx, (filename, page) in enumerate(top_docs, start=1):
            sources_list.append(f"  * [{idx}] `{filename}` (Page {page})")
            
        sources_text = "\n".join(sources_list) if sources_list else "  * No context chunks retrieved."
        
        query_details.append({
            "index": i + 1,
            "question": question,
            "answer": answer,
            "type": q_type,
            "sources": sources_text,
            "latency": latency,
            "faithfulness": faithfulness,
            "recall": context_recall,
            "precision": context_precision
        })
        
        # Save state after each successful query
        with open(state_file, "w") as f:
            json.dump({
                "all_faithfulness": all_faithfulness,
                "all_recall": all_recall,
                "all_precision": all_precision,
                "all_latency": all_latency,
                "query_details": query_details
            }, f)
        
    avg_faithfulness = np.mean(all_faithfulness) if all_faithfulness else 0.0
    avg_recall = np.mean(all_recall) if all_recall else 0.0
    avg_precision = np.mean(all_precision) if all_precision else 0.0
    p50_latency = np.percentile(all_latency, 50) if all_latency else 0.0
    p95_latency = np.percentile(all_latency, 95) if all_latency else 0.0
    
    # Calculate abstention rate
    abstentions = sum(1 for q in query_details if "does not contain sufficient information" in q["answer"])
    abstention_rate = abstentions / total if total > 0 else 0.0
    
    # Generate Markdown Results
    markdown = "# Production RAG Evaluation (Groq) & Harness Results\n\n"
    markdown += "## Aggregate Metrics\n\n"
    markdown += f"| Metric | Score / Value |\n"
    markdown += f"| :--- | :--- |\n"
    markdown += f"| **Total Queries** | {total} |\n"
    markdown += f"| **Abstention Rate** | {abstention_rate:.2%} ({abstentions}/{total}) |\n"
    markdown += f"| **Average Faithfulness (Groundedness)** | {avg_faithfulness:.2%} |\n"
    markdown += f"| **Average Context Recall** | {avg_recall:.2%} |\n"
    markdown += f"| **Average Context Precision** | {avg_precision:.2%} |\n"
    markdown += f"| **Latency p50** | {p50_latency:.2f}s |\n"
    markdown += f"| **Latency p95** | {p95_latency:.2f}s |\n\n"
    
    markdown += "---\n\n## Detailed Q&A Report (Executive View)\n\n"
    for q in query_details:
        markdown += (
            f"### {q['index']}. Question: {q['question']}\n\n"
            f"**Answer:**\n{q['answer']}\n\n"
            f"**Top Sources Consulted:**\n{q['sources']}\n\n"
            f"> *System Metrics ({q['type'].upper()}): Latency: {q['latency']:.2f}s | Faithfulness: {q['faithfulness']:.2%} | Context Recall: {q['recall']:.2%} | Context Precision: {q['precision']:.2%}*\n\n"
            f"---\n\n"
        )
        
    out_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "test_results_grok.md"
    )
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(markdown)
        
    print(f"\nEvaluation Complete! Aggregate results written to {out_path}")
    print(f"Avg Faithfulness: {avg_faithfulness:.2%}")
    print(f"Avg Recall: {avg_recall:.2%}")
    print(f"Avg Precision: {avg_precision:.2%}")
    print(f"p50 Latency: {p50_latency:.2f}s | p95 Latency: {p95_latency:.2f}s")
    
if __name__ == "__main__":
    run_eval()
