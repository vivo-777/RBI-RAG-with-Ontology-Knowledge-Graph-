import os, sys, pickle
sys.path.insert(0, os.path.dirname(__file__))
import config
from qdrant_client import QdrantClient
from qdrant_client.http import models

filename_to_remove = "6a38ce305640d_ET_AI_Hackathon_2026_Problem_Statements.pdf"
print(f"Removing {filename_to_remove} from Qdrant and BM25...")

db_path = os.getenv("QDRANT_DATA_PATH", os.path.abspath(os.path.join(os.path.dirname(__file__), "qdrant_data_v2")))
try:
    client = QdrantClient(path=db_path)
    # Remove from main collection
    client.delete(
        collection_name=config.QDRANT_COLLECTION,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="filename",
                        match=models.MatchValue(value=filename_to_remove),
                    )
                ]
            )
        )
    )
    # Remove from atomic collection
    client.delete(
        collection_name="rbi_atomic_propositions",
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="filename",
                        match=models.MatchValue(value=filename_to_remove),
                    )
                ]
            )
        )
    )
    print("Deleted from Qdrant.")
except Exception as e:
    print(f"Error deleting from Qdrant: {e}")

bm25_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "bm25_data.pkl"))
try:
    with open(bm25_path, 'rb') as f:
        data = pickle.load(f)
        
    corpus = data["corpus"]
    payloads = data["payloads"]
    
    new_corpus = []
    new_payloads = []
    
    for c, p in zip(corpus, payloads):
        if p.get("filename") != filename_to_remove:
            new_corpus.append(c)
            new_payloads.append(p)
            
    print(f"BM25 size before: {len(corpus)}, after: {len(new_corpus)}")
    
    from rank_bm25 import BM25Okapi
    from retrieval.sparse import sparse_index
    sparse_index.corpus = new_corpus
    sparse_index.payloads = new_payloads
    sparse_index.build()
    sparse_index.save(bm25_path)
    print("Deleted from BM25.")
except Exception as e:
    print(f"Error deleting from BM25: {e}")
