"""
Diagnostic script: For each failing query, show:
1. What the QueryClassifier labels it as
2. The top retrieved chunks (filename, rerank score, content excerpt)
3. Whether the expected source PDF appears in the results
"""
import json, os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
config.GROQ_DEFAULT_MODEL = "qwen2.5:7b-instruct"
config.GROQ_ESCALATION_MODEL = "qwen2.5:7b-instruct"

import dspy
from retrieval.fusion import hybrid_search
from retrieval.query_decomposer import QueryClassifier, expand_query
from generation.dspy_advisor import load_advisor

# Failing query IDs (1-indexed)
FAILING_IDS = [5, 8, 25, 28, 31, 33, 34, 35, 37, 38, 39, 40]

queries_path = os.path.join(os.path.dirname(__file__), "eval", "queries.json")
with open(queries_path) as f:
    all_queries = json.load(f)

advisor, lm = load_advisor("qwen2.5:7b-instruct")

print("=" * 100)
print("RETRIEVAL DIAGNOSIS FOR FAILING QUERIES")
print("=" * 100)

for qid in FAILING_IDS:
    q = all_queries[qid - 1]  # 0-indexed
    query_text = q["question"]
    expected_sources = q.get("expected_citations", [])
    
    print(f"\n{'='*100}")
    print(f"QUERY {qid}: {query_text}")
    print(f"Expected sources: {expected_sources}")
    print(f"{'='*100}")
    
    # 1. Classification
    expanded = expand_query(query_text)
    classifier = QueryClassifier()
    try:
        with dspy.context(lm=lm):
            category = classifier(query=expanded)
    except Exception as e:
        category = f"ERROR: {e}"
    print(f"  Classification: {category}")
    print(f"  Expanded query: {expanded}")
    
    # 2. Retrieval
    results = hybrid_search(expanded)
    print(f"  Retrieved {len(results)} chunks")
    
    found_expected = False
    for i, chunk in enumerate(results):
        payload = chunk["payload"]
        filename = payload.get("filename", "?")
        rerank_score = chunk.get("rerank_score", 0.0)
        content = payload.get("content", "")
        excerpt = content[:120].replace("\n", " ")
        
        is_match = any(exp.lower() in filename.lower() for exp in expected_sources)
        marker = " <<<< EXPECTED SOURCE" if is_match else ""
        if is_match:
            found_expected = True
        
        print(f"    [{i+1}] score={rerank_score:.4f} | {filename}{marker}")
        print(f"        {excerpt}...")
    
    if not found_expected:
        print(f"  *** WARNING: None of the expected source PDFs were found in retrieved chunks! ***")
    print()

print("\n" + "=" * 100)
print("DIAGNOSIS COMPLETE")
