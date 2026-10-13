import json
from retrieval.fusion import hybrid_search

def run_eval():
    print("Running Eval Suite...")
    
    with open("eval/gold_set.jsonl", "r") as f:
        cases = [json.loads(line) for line in f]
        
    for case in cases:
        query = case["query"]
        expected_citations = case["expected_citations"]
        
        # Mock retrieval for eval
        results = hybrid_search(query)
        retrieved_files = [res["payload"].get("filename") for res in results]
        
        hit = any(expected in retrieved_files for expected in expected_citations)
        print(f"Query: {query} | Hit: {hit}")

if __name__ == "__main__":
    run_eval()
