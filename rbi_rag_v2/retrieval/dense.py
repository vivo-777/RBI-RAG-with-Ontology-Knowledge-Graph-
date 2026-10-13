from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer
import config
import os

import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
embed_model = SentenceTransformer(config.EMBEDDING_MODEL, device=device)

db_path = os.getenv("QDRANT_DATA_PATH", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "qdrant_data_v2")))
client = QdrantClient(path=db_path)

def retrieve_dense(query: str, top_k: int = config.TOP_K_DENSE, domain_filter: str = None):
    """
    Queries Qdrant using dense embeddings.
    """
    query_vector = embed_model.encode(query).tolist()
    
    query_filter = None
    if domain_filter:
        query_filter = Filter(
            must=[
                FieldCondition(
                    key="domains",
                    match=MatchValue(value=domain_filter)
                )
            ]
        )
        
    # In a real app, ensure the collection exists before querying.
    try:
        results = client.search(
            collection_name=config.QDRANT_COLLECTION,
            query_vector=query_vector,
            query_filter=query_filter,
            limit=top_k
        )
        
        return [
            {
                "id": hit.id,
                "score": hit.score,
                "payload": hit.payload
            }
            for hit in results
        ]
    except Exception as e:
        # Collection might not exist yet
        return []

ATOMIC_COLLECTION = "rbi_atomic_propositions"

def retrieve_atomic(query: str, top_k: int = 15) -> list:
    """Search atomic propositions, return parent chunk payloads."""
    query_vector = embed_model.encode(query).tolist()
    try:
        results = client.search(
            collection_name=ATOMIC_COLLECTION,
            query_vector=query_vector,
            limit=top_k
        )
        # Expand atomic hits back to parent chunks
        seen_parents = set()
        parent_results = []
        for hit in results:
            parent_id = hit.payload.get("parent_id")
            if parent_id and parent_id not in seen_parents:
                seen_parents.add(parent_id)
                parent_results.append({
                    "id": parent_id,
                    "score": hit.score,
                    "payload": {
                        "chunk_id": parent_id,
                        "content": hit.payload.get("parent_content", ""),
                        "filename": hit.payload.get("filename", ""),
                        "page_number": hit.payload.get("page_number", 1),
                        "domains": hit.payload.get("domains", []),
                    }
                })
        return parent_results
    except Exception:
        return []
