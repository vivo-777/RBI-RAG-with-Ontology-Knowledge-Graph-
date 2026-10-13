import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from retrieval.fusion import hybrid_search
import config

config.MAX_CONTEXT_TOKENS = 4000

query = "Which specific circular regarding Perpetual Debt Instruments (PDI) in Additional Tier 1 Capital was repealed on September 29, 2025?"

retrieved = hybrid_search(query)
print(f"Retrieved {len(retrieved)} chunks.")
for i, chunk in enumerate(retrieved[:5]):
    payload = chunk["payload"]
    content = payload.get("content", "")
    print(f"\n--- Chunk {i+1} ---")
    print(f"Filename: {payload.get('filename')}")
    print(f"Score: {chunk.get('rerank_score')}")
    print(f"Content length: {len(content.split())} words")
    print(f"Content excerpt: {content[:300]}...")
