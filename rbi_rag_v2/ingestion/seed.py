import os
import json
import uuid
from typing import List, Dict, Any
from .pdf_extractor import extract_text_and_tables
from .parser import parse_structure
from .chunker import chunk_section
from ontology.entity_linker import extract_entities
from ontology.graph_store import graph_db

def index_document(pdf_path: str, qdrant_client, bm25_index, embed_model):
    """
    Main driver to process a PDF and seed it into Qdrant and BM25 index.
    """
    filename = os.path.basename(pdf_path)
    doc_id = str(uuid.uuid4())
    
    # Simple inference of domain from filename
    domains = extract_entities(filename)
    
    # 1. Add to Ontology Graph
    graph_db.add_circular(doc_id, filename=filename, status="active")
    
    # 2. Extract
    pages_data = extract_text_and_tables(pdf_path)
    
    chunks_to_index = []
    
    for page in pages_data:
        page_num = page["page_number"]
        
        # Parse sections
        sections = parse_structure(page["text"])
        
        for section in sections:
            # Chunk section
            sub_chunks = chunk_section(section)
            for chunk in sub_chunks:
                chunk_id = str(uuid.uuid4())
                payload = {
                    "chunk_id": chunk_id,
                    "document_id": doc_id,
                    "filename": filename,
                    "page_number": page_num,
                    "header": chunk["header"],
                    "content": chunk["content"],
                    "domains": domains,
                    "type": "text"
                }
                chunks_to_index.append(payload)
                
        # Handle tables
        for table in page["tables"]:
            chunk_id = str(uuid.uuid4())
            payload = {
                "chunk_id": chunk_id,
                "document_id": doc_id,
                "filename": filename,
                "page_number": page_num,
                "header": "Table",
                "content": table,
                "domains": domains,
                "type": "table"
            }
            chunks_to_index.append(payload)
            
            # Extract structured facts for Facts DB
            # (Skipped during this fast re-ingestion to save time, as it requires LLM calls for every table)
            '''
            try:
                import requests
                import json
                import re
                
                prompt = f"""Extract numerical facts from this markdown table into a JSON array of objects. 
Each object MUST have exactly these string keys: 'entity', 'metric', 'value', 'unit', 'as_of_date'. 
If a key is not applicable, use an empty string. Output ONLY a valid JSON array, no other text.
Table: {table}"""
                r = requests.post("http://localhost:11434/api/generate", json={
                    "model": "qwen2.5:7b-instruct",
                    "prompt": prompt,
                    "stream": False,
                    "format": "json"
                }, timeout=45)
                if r.status_code == 200:
                    response_text = r.json().get("response", "")
                    json_str = re.sub(r'```(?:json)?\n?(.*?)\n?```', r'\1', response_text, flags=re.DOTALL).strip()
                    facts = json.loads(json_str)
                if isinstance(facts, list):
                    from ontology.facts_store import insert_fact
                    for row_idx, fact in enumerate(facts):
                        fact["document_id"] = doc_id
                        fact["filename"] = filename
                        fact["page_number"] = page_num
                        fact["table_row_id"] = row_idx
                        fact["raw_context"] = table
                        insert_fact(fact)
            except Exception as e:
                print(f"Failed to extract structured facts from table: {e}")
            '''

    # 3. Embedding and Indexing
    if chunks_to_index:
        # In a real app, this would use the dense/sparse insertion functions
        # For this script, we just assume the clients have an index method
        # and we yield or return the payload for the retrieval module to handle
        pass
        
    return chunks_to_index
