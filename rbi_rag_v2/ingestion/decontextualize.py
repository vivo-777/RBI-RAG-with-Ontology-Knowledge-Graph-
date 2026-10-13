import os
import sys
import uuid
import json
import requests
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any

# Add parent dir to path so we can import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from ingestion.seed import extract_text_and_tables, parse_structure, chunk_section, extract_entities
from retrieval.dense import client as qdrant_client, embed_model
from retrieval.sparse import sparse_index
from qdrant_client.http.models import Distance, VectorParams, PointStruct

QDRANT_ATOMIC_COLLECTION = "rbi_atomic_propositions"
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:7b-instruct"

OLLAMA_IS_UP = False
try:
    if requests.get("http://localhost:11434/", timeout=1).status_code == 200:
        OLLAMA_IS_UP = True
except:
    pass

def decontextualize_chunk(chunk_text: str, context_texts: List[str]) -> Dict[str, Any]:
    if not getattr(config, "ENABLE_FULL_DECONTEXTUALIZATION", False) or not OLLAMA_IS_UP:
        import re
        sentences = [s.strip() for s in re.split(r'(?<=[.!?]) +', chunk_text) if s.strip()]
        if not sentences:
            sentences = [chunk_text]
        return {
            "decontextualized_text": chunk_text,
            "propositions": sentences
        }

    context_str = "\n\n".join([f"Context chunk {idx+1}:\n{text}" for idx, text in enumerate(context_texts)])
    
    prompt = f"""You are a regulatory compliance assistant.
Given preceding context paragraphs from the same document (if any), followed by the current paragraph, perform two tasks:
1. Decontextualize the current paragraph: Resolve all vague pronouns or references (e.g. "this circular", "the Bank", "such accounts", "they", "it") in the current paragraph using the context of the preceding paragraphs. Replace them with their fully named referents (e.g. "Reserve Bank of India", "Priority Sector Lending guidelines", "commercial banks").
2. Decompose the decontextualized paragraph into a list of independent, self-contained atomic propositions. Each proposition must represent a single, clear regulatory fact/rule, and must fully name the subject, predicate, and object (no pronouns or relative references).

Output strictly as a JSON object matching this schema:
{{
  "decontextualized_text": "The fully decontextualized current paragraph",
  "propositions": [
    "Proposition 1 text",
    "Proposition 2 text"
  ]
}}
Do not include any other markdown formatting or conversational text outside of the JSON object.

{context_str}

Current paragraph:
{chunk_text}
"""
    try:
        r = requests.post(OLLAMA_URL, json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "format": "json"
        }, timeout=2)
        if r.status_code == 200:
            res_obj = r.json()
            response_text = res_obj.get("response", "")
            data = json.loads(response_text)
            if "decontextualized_text" in data and "propositions" in data:
                return data
    except Exception as e:
        print(f"Ollama decontextualization failed: {e}")
        
    import re
    sentences = [s.strip() for s in re.split(r'(?<=[.!?]) +', chunk_text) if s.strip()]
    return {
        "decontextualized_text": chunk_text,
        "propositions": sentences
    }

METADATA_MAP = {
    "04MCKCC03072017.pdf": {
        "issue_date": "2017-07-03",
        "effective_from": "2017-07-03",
        "superseded_by": "KCC Modified.pdf",
        "entity_type": ["Commercial Banks", "RRBs", "Cooperative Banks"],
    },
    "KCC Modified.pdf": {
        "issue_date": "2022-04-01",
        "effective_from": "2022-04-01",
        "superseded_by": None,
        "entity_type": ["Commercial Banks", "RRBs", "Cooperative Banks"],
    },
    "10MC02072019.pdf": {
        "issue_date": "2019-07-02",
        "effective_from": "2019-07-02",
        "superseded_by": None,
        "entity_type": ["Commercial Banks", "Cooperative Banks"],
    },
    "128MD66C4DDCB167C4DC9A5BD913570CB3D47 (1).pdf": {
        "issue_date": "2016-02-25",
        "effective_from": "2016-02-25",
        "superseded_by": None,
        "entity_type": ["All Regulated Entities"],
    },
    "128MD66C4DDCB167C4DC9A5BD913570CB3D47.pdf": {
        "issue_date": "2016-02-25",
        "effective_from": "2016-02-25",
        "superseded_by": None,
        "entity_type": ["All Regulated Entities"],
    },
    "34MD27062019.pdf": {
        "issue_date": "2019-06-27",
        "effective_from": "2019-06-27",
        "superseded_by": None,
        "entity_type": ["Commercial Banks"],
    },
    "56MD24072017E50D0ED63F9B4414AA756FF0FC72FB66.pdf": {
        "issue_date": "2017-07-24",
        "effective_from": "2017-07-24",
        "superseded_by": "msme_ammendments.pdf",
        "entity_type": ["Commercial Banks", "RRBs", "Cooperative Banks"],
    },
    "msme_ammendments.pdf": {
        "issue_date": "2020-06-26",
        "effective_from": "2020-06-26",
        "superseded_by": None,
        "entity_type": ["Commercial Banks", "RRBs", "Cooperative Banks"],
    },
    "BASEL III CAPITAL FRAMEWORK.pdf": {
        "issue_date": "2023-04-01",
        "effective_from": "2023-04-01",
        "superseded_by": None,
        "entity_type": ["Commercial Banks"],
    },
    "DAY NRLMS.pdf": {
        "issue_date": "2023-06-01",
        "effective_from": "2023-06-01",
        "superseded_by": None,
        "entity_type": ["Commercial Banks", "RRBs", "Cooperative Banks"],
    },
    "FID250215CIRNT.pdf": {
        "issue_date": "2015-02-25",
        "effective_from": "2015-02-25",
        "superseded_by": None,
        "entity_type": ["Commercial Banks", "RRBs", "Cooperative Banks"],
    },
    "Master-circular-on-lead-bank-scheme.pdf": {
        "issue_date": "2023-07-01",
        "effective_from": "2023-07-01",
        "superseded_by": None,
        "entity_type": ["Commercial Banks"],
    },
    "RBI-2024-2025-96_712202414454187.pdf": {
        "issue_date": "2024-01-01",
        "effective_from": "2024-01-01",
        "superseded_by": None,
        "entity_type": ["Commercial Banks", "Cooperative Banks"],
    },
    "RBI-PSL-MASTER-DIRECTIONS-24-03-25.pdf": {
        "issue_date": "2025-03-24",
        "effective_from": "2025-03-24",
        "superseded_by": None,
        "entity_type": ["Commercial Banks", "RRBs", "Cooperative Banks", "Small Finance Banks"],
    },
    "RFML30012025.pdf": {
        "issue_date": "2025-01-30",
        "effective_from": "2025-01-30",
        "superseded_by": None,
        "entity_type": ["Commercial Banks"],
    },
    "Sep72022_RBI_RBI_Review_of_Prudential_Norms_in_regard_to_Risk_Weights_for_Exposures_guaranteed_by_Credit_Guarantee_Schemes.PDF": {
        "issue_date": "2022-09-07",
        "effective_from": "2022-09-07",
        "superseded_by": None,
        "entity_type": ["Commercial Banks"],
    },
    "dri rbi.pdf": {
        "issue_date": "2021-04-01",
        "effective_from": "2021-04-01",
        "superseded_by": None,
        "entity_type": ["Commercial Banks"],
    },
}

def get_metadata(filename: str, header: str) -> Dict[str, Any]:
    meta = METADATA_MAP.get(filename, {
        "issue_date": "2023-01-01",
        "effective_from": "2023-01-01",
        "superseded_by": None,
        "entity_type": ["All Regulated Entities"]
    })
    return {
        "source_circular": filename,
        "issue_date": meta["issue_date"],
        "effective_from": meta["effective_from"],
        "superseded_by": meta["superseded_by"],
        "entity_type": meta["entity_type"],
        "section_path": header or "General"
    }

def init_qdrant_atomic(recreate=True):
    # Recreate parent collection
    from ingestion.run_ingestion import init_qdrant
    init_qdrant(recreate=recreate)
    
    # Recreate atomic collection
    if recreate:
        try:
            qdrant_client.delete_collection(QDRANT_ATOMIC_COLLECTION)
            print(f"Deleted existing Qdrant atomic collection: {QDRANT_ATOMIC_COLLECTION}")
        except Exception as e:
            print(f"No existing atomic collection to delete: {e}")
    try:
        qdrant_client.get_collection(QDRANT_ATOMIC_COLLECTION)
        print(f"Collection {QDRANT_ATOMIC_COLLECTION} already exists.")
    except Exception:
        print(f"Creating Qdrant atomic collection: {QDRANT_ATOMIC_COLLECTION}...")
        dim = embed_model.get_sentence_embedding_dimension()
        qdrant_client.create_collection(
            collection_name=QDRANT_ATOMIC_COLLECTION,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )




PRIMARY_EVAL_PDFS = {
    "BASEL III CAPITAL FRAMEWORK.pdf",
    "56MD24072017E50D0ED63F9B4414AA756FF0FC72FB66.pdf",
    "04MCKCC03072017.pdf"
}

ENABLE_FULL_DECONTEXTUALIZATION = True

def process_pdf(pdf_path: str) -> tuple:
    filename = os.path.basename(pdf_path)
    doc_id = str(uuid.uuid4())
    domains = extract_entities(filename)
    pages_data = extract_text_and_tables(pdf_path)
    
    raw_chunks_info = []
    for page in pages_data:
        page_num = page["page_number"]
        sections = parse_structure(page["text"])
        
        for section in sections:
            sub_chunks = chunk_section(section)
            for chunk in sub_chunks:
                raw_chunks_info.append({
                    "type": "text",
                    "page_number": page_num,
                    "header": chunk["header"],
                    "content": chunk["content"]
                })
                
        for table in page["tables"]:
            raw_chunks_info.append({
                "type": "table",
                "page_number": page_num,
                "header": "Table",
                "content": table
            })
            
    # Prepare sliding window tasks
    tasks = []
    for idx, item in enumerate(raw_chunks_info):
        if item["type"] == "table":
            preceding = []
        else:
            # Gather up to 2 preceding text chunks as context
            preceding = []
            for j in range(max(0, idx - 2), idx):
                if raw_chunks_info[j]["type"] == "text":
                    preceding.append(raw_chunks_info[j]["content"])
        tasks.append((item, preceding))
        
    parent_chunks = []
    atomic_units = []
    
    # Run decontextualization
    results = [None] * len(tasks)
    if ENABLE_FULL_DECONTEXTUALIZATION or filename in PRIMARY_EVAL_PDFS:
        print(f"Submitting {len(tasks)} decontextualization tasks for {filename}...")
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {
                executor.submit(decontextualize_chunk, item["content"], preceding): (item, idx)
                for idx, (item, preceding) in enumerate(tasks)
            }
            for future in concurrent.futures.as_completed(futures):
                item, idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    print(f"Error on chunk {idx}: {e}")
                    results[idx] = {
                        "decontextualized_text": item["content"],
                        "propositions": [item["content"]]
                    }
    else:
        print(f"Skipping LLM decontextualization for non-eval PDF {filename} (indexing raw chunks directly)...")
        for idx, (item, preceding) in enumerate(tasks):
            import re
            sentences = [s.strip() for s in re.split(r'(?<=[.!?]) +', item["content"]) if s.strip()]
            results[idx] = {
                "decontextualized_text": item["content"],
                "propositions": sentences
            }
                
    # Build outputs
    for idx, item in enumerate(raw_chunks_info):
        res = results[idx]
        decontext_text = res.get("decontextualized_text", item["content"])
        propositions = res.get("propositions", [item["content"]])
        
        chunk_id = str(uuid.uuid4())
        parent_payload = {
            "chunk_id": chunk_id,
            "document_id": doc_id,
            "filename": filename,
            "page_number": item["page_number"],
            "header": item["header"],
            "content": decontext_text,
            "raw_content": item["content"],
            "domains": domains,
            "type": item["type"]
        }
        parent_chunks.append(parent_payload)
        
        meta = get_metadata(filename, item["header"])
        for prop in propositions:
            atomic_payload = {
                "atomic_id": str(uuid.uuid4()),
                "parent_id": chunk_id,
                "parent_content": decontext_text,
                "content": decontext_text,
                "proposition": prop,
                "filename": filename,
                "effective_from": meta["effective_from"],
                "page_number": item["page_number"],
                "domains": domains,
                **meta
            }
            atomic_units.append(atomic_payload)
            
    return parent_chunks, atomic_units

def run(recreate_db=True):
    circulars_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "circulars"))
    print(f"Target directory: {circulars_dir}")
    
    if not os.path.exists(circulars_dir):
        print(f"Directory {circulars_dir} not found!")
        return
        
    init_qdrant_atomic(recreate=recreate_db)
    
    pdfs = [f for f in os.listdir(circulars_dir) if f.lower().endswith(".pdf")]
    print(f"Found {len(pdfs)} PDFs. Initiating decontextualization...")
    
    hashes_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "processed_hashes.json"))
    from ingestion.dedup import get_file_sha256, get_pdf_text_hash
    
    if not recreate_db and os.path.exists(hashes_file):
        with open(hashes_file, "r") as f:
            processed_data = json.load(f)
    else:
        processed_data = {"file_hashes": [], "text_hashes": []}
        
    file_hashes = set(processed_data.get("file_hashes", []))
    text_hashes = set(processed_data.get("text_hashes", []))
    
    if recreate_db:
        file_hashes.clear()
        text_hashes.clear()
        sparse_index.corpus = []
        sparse_index.payloads = []
        sparse_index.bm25 = None
        
    # Process files
    for pdf in pdfs:
        pdf_path = os.path.join(circulars_dir, pdf)
        try:
            f_hash = get_file_sha256(pdf_path)
            t_hash = get_pdf_text_hash(pdf_path)
        except Exception as e:
            print(f"Failed hashing {pdf}: {e}")
            continue
            
        if f_hash in file_hashes or t_hash in text_hashes:
            print(f"Skipping duplicate: {pdf}")
            continue
            
        print(f"Processing decontextualization for {pdf}...")
        parents, atomics = process_pdf(pdf_path)
        
        # Upsert parent chunks
        parent_points = []
        for p in parents:
            vector = embed_model.encode(p["content"]).tolist()
            parent_points.append(PointStruct(id=p["chunk_id"], vector=vector, payload=p))
            sparse_index.add(p["content"], p)
        if parent_points:
            qdrant_client.upsert(collection_name=config.QDRANT_COLLECTION, points=parent_points)
            
        # Upsert atomic units
        atomic_points = []
        for a in atomics:
            vector = embed_model.encode(a["proposition"]).tolist()
            atomic_points.append(PointStruct(id=a["atomic_id"], vector=vector, payload=a))
            # Also index propositions in BM25 pointing to parent payload
            sparse_index.add(a["proposition"], a)
            
        if atomic_points:
            qdrant_client.upsert(collection_name=QDRANT_ATOMIC_COLLECTION, points=atomic_points)
            
        file_hashes.add(f_hash)
        text_hashes.add(t_hash)
        
    with open(hashes_file, "w") as f:
        json.dump({
            "file_hashes": list(file_hashes),
            "text_hashes": list(text_hashes)
        }, f, indent=2)
        
    print("Building BM25 Index...")
    sparse_index.build()
    bm25_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "bm25_data.pkl"))
    sparse_index.save(bm25_path)
    print("Ingestion decontextualization complete.")

if __name__ == "__main__":
    recreate_db = "--no-recreate" not in sys.argv
    run(recreate_db=recreate_db)
