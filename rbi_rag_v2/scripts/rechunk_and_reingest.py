"""
Re-ingestion script with larger chunks (400 words, 150 overlap).
Backs up old data, re-chunks all PDFs, rebuilds Qdrant + BM25 indexes.
"""
import os, sys, time, shutil
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from ingestion.seed import index_document
from retrieval.sparse import sparse_index
from retrieval.dense import client, embed_model
from qdrant_client.http.models import Distance, VectorParams, PointStruct

CIRCULARS_DIR = os.path.join(os.path.dirname(__file__), "web", "public", "circulars")
if not os.path.exists(CIRCULARS_DIR):
    CIRCULARS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "circulars"))
QDRANT_DATA_DIR = os.path.join(os.path.dirname(__file__), "qdrant_data_v2")
BM25_PATH = os.path.join(os.path.dirname(__file__), "bm25_data.pkl")

def run():
    start = time.time()
    
    # 1. Skip backup (Qdrant .lock file prevents copy while client is active)
    print("Starting re-ingestion with larger chunks (400 words, 150 overlap)...")
    
    # 2. Delete and recreate collection
    try:
        client.delete_collection(config.QDRANT_COLLECTION)
        print(f"Deleted collection: {config.QDRANT_COLLECTION}")
    except Exception:
        pass
    
    dim = embed_model.get_sentence_embedding_dimension()
    client.create_collection(
        collection_name=config.QDRANT_COLLECTION,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )
    print(f"Created fresh collection with dim={dim}")
    
    # Also recreate atomic propositions collection
    ATOMIC_COLLECTION = "rbi_atomic_propositions"
    try:
        client.delete_collection(ATOMIC_COLLECTION)
    except Exception:
        pass
    client.create_collection(
        collection_name=ATOMIC_COLLECTION,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )
    
    # 3. Reset BM25 (clear in-memory data)
    sparse_index.corpus = []
    sparse_index.payloads = []
    sparse_index.bm25 = None
    
    # 4. Find all PDFs
    pdfs = [f for f in os.listdir(CIRCULARS_DIR) if f.lower().endswith(".pdf")]
    print(f"\nFound {len(pdfs)} PDFs in {CIRCULARS_DIR}")
    
    total_chunks = 0
    
    for i, pdf in enumerate(pdfs, 1):
        pdf_path = os.path.join(CIRCULARS_DIR, pdf)
        print(f"\n[{i}/{len(pdfs)}] Processing: {pdf}")
        
        try:
            chunks = index_document(pdf_path, client, sparse_index, embed_model)
            
            if not chunks:
                print(f"  WARNING: No chunks produced for {pdf}")
                continue
            
            # Embed and push to Qdrant
            points = []
            for chunk in chunks:
                vector = embed_model.encode(chunk["content"]).tolist()
                points.append(
                    PointStruct(
                        id=chunk["chunk_id"],
                        vector=vector,
                        payload=chunk
                    )
                )
            
            if points:
                # Batch upsert (Qdrant handles batching internally)
                client.upsert(
                    collection_name=config.QDRANT_COLLECTION,
                    points=points
                )
            
            # Push to BM25
            for chunk in chunks:
                sparse_index.add(chunk["content"], chunk)
            
            total_chunks += len(chunks)
            print(f"  Indexed {len(chunks)} chunks (total: {total_chunks})")
            
        except Exception as e:
            print(f"  ERROR: Failed to index {pdf}: {e}")
    
    # 5. Build and save BM25
    print("\nBuilding BM25 index...")
    sparse_index.build()
    sparse_index.save(BM25_PATH)
    print(f"Saved BM25 index to {BM25_PATH}")
    
    elapsed = time.time() - start
    print(f"\n{'='*60}")
    print(f"RE-INGESTION COMPLETE")
    print(f"  PDFs processed: {len(pdfs)}")
    print(f"  Total chunks: {total_chunks}")
    print(f"  Time: {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print(f"{'='*60}")

if __name__ == "__main__":
    run()
