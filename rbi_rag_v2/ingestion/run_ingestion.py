import os
import sys

# Add parent dir to path so we can import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingestion.seed import index_document
from retrieval.sparse import sparse_index
from retrieval.dense import client, embed_model
from qdrant_client.http.models import Distance, VectorParams
import config

def init_qdrant(recreate=True):
    if recreate:
        try:
            client.delete_collection(config.QDRANT_COLLECTION)
            print(f"Deleted existing Qdrant collection: {config.QDRANT_COLLECTION}")
        except Exception:
            pass
    try:
        client.get_collection(config.QDRANT_COLLECTION)
        print("Collection already exists.")
    except Exception:
        print("Creating Qdrant collection...")
        # Get embedding dimension
        dim = embed_model.get_sentence_embedding_dimension()
        client.create_collection(
            collection_name=config.QDRANT_COLLECTION,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )

def run():
    circulars_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "circulars"))
    print(f"Target directory: {circulars_dir}")
    
    if not os.path.exists(circulars_dir):
        print(f"Directory {circulars_dir} not found!")
        return
        
    init_qdrant()
    
    pdfs = [f for f in os.listdir(circulars_dir) if f.lower().endswith(".pdf")]
    print(f"Found {len(pdfs)} PDFs.")
    
    for pdf in pdfs:
        pdf_path = os.path.join(circulars_dir, pdf)
        print(f"Indexing {pdf}...")
        try:
            chunks = index_document(pdf_path, client, sparse_index, embed_model)
            
            # Push to Qdrant
            from qdrant_client.http.models import PointStruct
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
                client.upsert(
                    collection_name=config.QDRANT_COLLECTION,
                    points=points
                )
                
            # Push to BM25
            for chunk in chunks:
                sparse_index.add(chunk["content"], chunk)
                
        except Exception as e:
            print(f"Failed to index {pdf}: {e}")
            
    print("Building BM25 Index...")
    sparse_index.build()
    
    bm25_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "bm25_data.pkl"))
    sparse_index.save(bm25_path)
    print(f"Saved BM25 index to {bm25_path}")
    print("Ingestion complete.")

if __name__ == "__main__":
    run()
