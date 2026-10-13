import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingestion.seed import extract_text_and_tables as normal_extract, parse_structure, chunk_section, extract_entities
from ingestion.ocr_extractor import extract_text_with_ocr
from retrieval.sparse import sparse_index
from retrieval.dense import client, embed_model
import config
import uuid
from ontology.graph_store import graph_db

def needs_ocr(pdf_path: str) -> bool:
    """Check if the PDF requires OCR by trying normal extraction and counting text length."""
    pages = normal_extract(pdf_path)
    total_text = sum(len(p.get("text", "").strip()) for p in pages)
    # If the entire PDF has less than 200 characters of text, it's likely scanned.
    return total_text < 200

def index_ocr_document(pdf_path: str):
    filename = os.path.basename(pdf_path)
    doc_id = str(uuid.uuid4())
    domains = extract_entities(filename)
    
    graph_db.add_circular(doc_id, filename=filename, status="active")
    
    print(f"Running EasyOCR extraction for {filename}...")
    pages_data = extract_text_with_ocr(pdf_path)
    
    chunks_to_index = []
    
    for page in pages_data:
        page_num = page["page_number"]
        sections = parse_structure(page["text"])
        
        for section in sections:
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
            
            # Use Ollama for table extraction just like normal ingestion
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

    # Push to Qdrant
    if chunks_to_index:
        from qdrant_client.http.models import PointStruct
        points = []
        for chunk in chunks_to_index:
            vector = embed_model.encode(chunk["content"]).tolist()
            points.append(
                PointStruct(
                    id=chunk["chunk_id"],
                    vector=vector,
                    payload=chunk
                )
            )
        client.upsert(
            collection_name=config.QDRANT_COLLECTION,
            points=points
        )
        # Push to BM25
        for chunk in chunks_to_index:
            sparse_index.add(chunk["content"], chunk)
            
    return len(chunks_to_index)

def run():
    circulars_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "circulars"))
    pdfs = [f for f in os.listdir(circulars_dir) if f.lower().endswith(".pdf")]
    
    ocr_pdfs = []
    print("Checking which PDFs need OCR...")
    for pdf in pdfs:
        pdf_path = os.path.join(circulars_dir, pdf)
        if needs_ocr(pdf_path):
            ocr_pdfs.append(pdf_path)
            
    print(f"Found {len(ocr_pdfs)} PDFs requiring OCR.")
    
    for pdf_path in ocr_pdfs:
        print(f"Indexing {os.path.basename(pdf_path)} with OCR...")
        try:
            chunks_count = index_ocr_document(pdf_path)
            print(f"Indexed {chunks_count} chunks from {os.path.basename(pdf_path)}")
        except Exception as e:
            print(f"Failed to index {os.path.basename(pdf_path)}: {e}")
            
    print("Re-saving BM25 Index...")
    bm25_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "bm25_data.pkl"))
    sparse_index.build()
    sparse_index.save(bm25_path)
    print("OCR Ingestion complete.")

if __name__ == "__main__":
    run()
