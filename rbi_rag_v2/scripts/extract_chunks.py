import json
import os

state_file = 'eval/eval_state.json'
output_file = 'chunks_sent_to_llm.md'

if not os.path.exists(state_file):
    print(f"Error: {state_file} not found.")
    exit(1)

with open(state_file, 'r', encoding='utf-8') as f:
    state = json.load(f)

results = state.get('results', [])

with open(output_file, 'w', encoding='utf-8') as f:
    f.write("# RAG Context Chunks Sent to the LLM\n\n")
    f.write("This file displays the actual text chunks retrieved from Qdrant and fed into the Groq 70B LLM for the first 5 queries during the evaluation phase.\n\n")
    
    for i, res in enumerate(results[:5]):
        query = res.get('query', '')
        f.write(f"## Query {i+1}: {query}\n\n")
        
        chunks = res.get('retrieved_chunks', [])
        if not chunks:
            f.write("> No context chunks were retrieved for this query.\n\n")
        
        for j, chunk in enumerate(chunks):
            filename = chunk.get('payload', {}).get('filename', 'Unknown_File')
            page = chunk.get('payload', {}).get('page_number', 'N/A')
            content = chunk.get('payload', {}).get('content', '')
            
            f.write(f"### Chunk {j+1} (Source: {filename}, Page: {page})\n")
            f.write("`	ext\n")
            f.write(f"{content}\n")
            f.write("`\n\n")
            
        f.write("---\n\n")

print("Created chunks_sent_to_llm.md successfully.")
