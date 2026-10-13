import json
import time
import os
import sys
import logging

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
# Override config for this run to use 8B model
config.GROQ_DEFAULT_MODEL = "llama-3.1-8b-instant"
config.GROQ_ESCALATION_MODEL = "llama-3.1-8b-instant"

from generation.dspy_advisor import generate_dspy_answer

logging.basicConfig(level=logging.INFO)

def run_40_eval():
    queries_path = os.path.join(os.path.dirname(__file__), "eval", "queries.json")
    with open(queries_path, "r") as f:
        queries = json.load(f)[:40]
        
    out_path = os.path.join(os.path.dirname(__file__), "groq_40_results.json")
    
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
        
        retries = 0
        while True:
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
                    
                # Artificial delay to help avoid immediate TPM limits on next query
                print("Sleeping for 15 seconds to respect Groq rate limits...")
                time.sleep(15)
                break
                
            except Exception as e:
                err_str = str(e).lower()
                if "rate limit" in err_str or "429" in err_str or "too many requests" in err_str:
                    retries += 1
                    if retries > 2:
                        print(f"\nFATAL: Rate limit persists after waiting. Your API key might be exhausted for the day.")
                        print(f"Update GROQ_API_KEY in .env and run this script again to resume from query {i+1}.")
                        sys.exit(1)
                    print(f"RATE LIMIT HIT (Groq TPM budget exhausted). Sleeping for 60 seconds before retrying...")
                    time.sleep(60)
                else:
                    print(f"Fatal Error on query {query_text}: {e}")
                    # Save what we have and exit
                    sys.exit(1)

    print(f"\nFinished evaluating 40 queries. Results saved to {out_path}")

if __name__ == "__main__":
    run_40_eval()
