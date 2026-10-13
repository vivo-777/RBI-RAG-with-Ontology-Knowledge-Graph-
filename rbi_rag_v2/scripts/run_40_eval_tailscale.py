import json
import time
import os
import sys
import logging

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
# Override config for this run to use LM Studio via Tailscale
# LM Studio ignores the model name if one is already loaded in the GUI.
config.GROQ_DEFAULT_MODEL = "qwen2.5:7b-instruct"
config.GROQ_ESCALATION_MODEL = "qwen2.5:7b-instruct"
config.MAX_CONTEXT_TOKENS = 11000  # We can use large context on local GPU!

from generation.dspy_advisor import generate_dspy_answer

logging.basicConfig(level=logging.INFO)

def run_40_eval():
    queries_path = os.path.join(os.path.dirname(__file__), "eval", "queries.json")
    with open(queries_path, "r") as f:
        queries = json.load(f)[:40]
        
    out_path = os.path.join(os.path.dirname(__file__), "tailscale_40_results.json")
    
    # Load existing results if any
    results = []
    if os.path.exists(out_path):
        with open(out_path, "r") as f:
            try:
                results = json.load(f)
                print(f"Resuming from query {len(results)+1}...")
            except json.JSONDecodeError:
                pass
    
    start_index = len(results)
    
    for i in range(start_index, len(queries)):
        q = queries[i]
        query_text = q["question"]
        print(f"\n[{i+1}/40] Evaluating: {query_text}")
        start = time.time()
        
        try:
            dspy_result = generate_dspy_answer(query_text)
            answer = dspy_result.get("answer", "")
            
            print(f"--> Answer: {answer}")
            print(f"--> Latency: {time.time() - start:.2f}s")
            
            results.append({
                "id": q.get("id", i+1),
                "query": query_text,
                "answer": answer,
                "latency": time.time() - start
            })
            
            # Save incrementally
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=4)
                
        except Exception as e:
            print(f"Fatal Error on query {query_text}: {e}")
            sys.exit(1)

    print(f"\nFinished evaluating 40 queries. Results saved to {out_path}")

if __name__ == "__main__":
    run_40_eval()
