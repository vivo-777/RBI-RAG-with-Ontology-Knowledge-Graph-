import json
import time
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
# Override config for this local run
config.GROQ_DEFAULT_MODEL = "qwen2.5:7b-instruct"
config.GROQ_ESCALATION_MODEL = "qwen2.5:7b-instruct"

from generation.dspy_advisor import generate_dspy_answer
from generation.citation_verifier import verify_response

def run_20_eval():
    queries_path = os.path.join(os.path.dirname(__file__), "eval", "queries.json")
    with open(queries_path, "r") as f:
        queries = json.load(f)[:20]
        
    results = []
    
    for i, q in enumerate(queries, start=1):
        query_text = q["question"]
        print(f"[{i}/20] Evaluating: {query_text}")
        start = time.time()
        
        try:
            dspy_result = generate_dspy_answer(query_text)
            answer = dspy_result.get("answer", "")
            
            print(f"--> Answer: {answer}")
            print(f"--> Latency: {time.time() - start:.2f}s\n")
            
            results.append({
                "query": query_text,
                "answer": answer,
                "latency": time.time() - start
            })
            
        except Exception as e:
            print(f"Error on query {query_text}: {e}\n")
            
    out_path = os.path.join(os.path.dirname(__file__), "qwen_20_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    print(f"Saved results to {out_path}")

if __name__ == "__main__":
    run_20_eval()
