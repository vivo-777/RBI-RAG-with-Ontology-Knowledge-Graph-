import json
import time
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from generation.dspy_advisor import generate_dspy_answer
from generation.citation_verifier import verify_response

def run_eval():
    queries_path = os.path.join(os.path.dirname(__file__), "queries.json")
    with open(queries_path, "r") as f:
        queries = json.load(f)
        
    results = []
    
    # Run pipeline for each query
    for q in queries:
        query_text = q["question"]
        print(f"Evaluating: {query_text}")
        start = time.time()
        
        try:
            dspy_result = generate_dspy_answer(query_text)
            retrieved = dspy_result.get("retrieved_chunks", [])
            answer = dspy_result.get("answer", "")
            latency = dspy_result.get("latency", 0.0)
            model_used = dspy_result.get("model", "unknown")
            votes = dspy_result.get("votes", [])
            
            if not retrieved:
                answer = "The provided regulatory context does not contain sufficient information to answer this query definitively."
                accuracy = 0.0
                claim_labels = []
            else:
                context_texts = [c["payload"].get("content", "") for c in retrieved]
                verification = verify_response(answer, context_texts)
                accuracy = verification["accuracy"]
                claim_labels = verification["claim_labels"]
                
                # Apply voting logic
                if votes:
                    vote_text = " ".join(votes).lower()
                    for claim_data in claim_labels:
                        if claim_data["label"] == "supported":
                            import re
                            numbers = set(re.findall(r'\b\d+(?:\.\d+)?\b', claim_data["claim"]))
                            for num in numbers:
                                if str(num) not in vote_text:
                                    claim_data["label"] = "partially_supported"
                                    # demote accuracy
                                    accuracy = max(0.0, accuracy - (0.5 / len(claim_labels)))
                                    break
                
            results.append({
                "query": query_text,
                "retrieved_doc_ids": [c["id"] for c in retrieved],
                "final_answer": answer,
                "claim_labels": claim_labels,
                "escalation": model_used == config.GROQ_ESCALATION_MODEL and model_used != config.GROQ_DEFAULT_MODEL,
                "latency": latency,
                "accuracy": accuracy,
                "type": q["type"]
            })
            
        except Exception as e:
            print(f"Error on query {query_text}: {e}")
            
    if not results:
        print("No results to summarize.")
        return

    # Compute aggregate metrics
    total_queries = len(results)
    mean_latency = sum(r["latency"] for r in results) / total_queries
    sorted_latencies = sorted([r["latency"] for r in results])
    p95_index = int(0.95 * total_queries)
    if p95_index >= total_queries:
        p95_index = total_queries - 1
    p95_latency = sorted_latencies[p95_index]
    mean_accuracy = sum(r["accuracy"] for r in results) / total_queries
    
    # Abstention check for out-of-context
    out_of_context = [r for r in results if r["type"] == "out_of_context"]
    abstention_count = sum(1 for r in out_of_context if "does not contain sufficient information" in r["final_answer"])
    abstention_rate = abstention_count / len(out_of_context) if out_of_context else 1.0
    
    # Generate test_results.md
    md_content = f"# Pipeline Evaluation Results\n\n"
    md_content += f"## Aggregate Metrics\n"
    md_content += f"- **Total Queries Evaluated:** {total_queries}\n"
    md_content += f"- **Mean Citation Accuracy (Faithfulness):** {mean_accuracy:.2%}\n"
    md_content += f"- **Correct Abstention Rate:** {abstention_rate:.2%}\n"
    md_content += f"- **Mean Latency:** {mean_latency:.2f}s\n"
    md_content += f"- **p95 Latency:** {p95_latency:.2f}s\n\n"
    
    md_content += f"## Per-Query Results\n\n"
    md_content += "| Query | Retrieved Docs | Escalation | Latency | Accuracy | Answer Snippet |\n"
    md_content += "|---|---|---|---|---|---|\n"
    for r in results:
        snippet = r["final_answer"].replace('\n', '<br>')
        docs = len(r["retrieved_doc_ids"])
        escalation = "Yes" if r["escalation"] else "No"
        md_content += f"| {r['query']} | {docs} docs | {escalation} | {r['latency']:.2f}s | {r['accuracy']:.2%} | {snippet} |\n"
        
    out_path = os.path.join(os.path.dirname(__file__), "..", "test_results.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"Evaluation complete. Results saved to {out_path}")

if __name__ == "__main__":
    run_eval()
