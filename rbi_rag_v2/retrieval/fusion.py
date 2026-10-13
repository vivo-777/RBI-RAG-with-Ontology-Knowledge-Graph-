from typing import List, Dict, Any
from sentence_transformers import CrossEncoder
import config
from .dense import retrieve_dense, retrieve_atomic
from .sparse import sparse_index
from ontology.graph_store import graph_db
from retrieval.hyde import generate_hypothetical_document
import torch
import hashlib

device = "cuda" if torch.cuda.is_available() else "cpu"
reranker = CrossEncoder(config.RERANKER_MODEL, device=device)

def apply_supersession_flag(results: List[Dict]) -> List[Dict]:
    for res in results:
        doc_id = res["payload"].get("document_id")
        if doc_id:
            active_doc_id = graph_db.get_active_circular(doc_id)
            if active_doc_id and active_doc_id != doc_id:
                res["payload"]["superseded_by"] = active_doc_id
                if "[WARNING: SUPERSEDED" not in res["payload"]["content"]:
                    res["payload"]["content"] = f"[WARNING: SUPERSEDED BY {active_doc_id}]\n" + res["payload"]["content"]
    return results

def rrf(dense_results: List[Dict], sparse_results: List[Dict], query: str = "", k: int = config.RRF_K) -> List[Dict]:
    scores = {}
    docs = {}
    
    for rank, res in enumerate(dense_results, 1):
        doc_id = res["id"]
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
        docs[doc_id] = res["payload"]
        
    for rank, res in enumerate(sparse_results, 1):
        doc_id = res["id"]
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
        docs[doc_id] = res["payload"]
        
    q_lower = query.lower()
    for doc_id, payload in docs.items():
        domains = payload.get("domains", [])
        if domains:
            for d in domains:
                if d and d.lower() in q_lower:
                    scores[doc_id] *= 1.2
                    break
                
    sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [{"id": doc_id, "score": score, "payload": docs[doc_id]} for doc_id, score in sorted_docs]

def rerank_unioned_results(query: str, fused_results: List[Dict]) -> List[Dict]:
    if not fused_results:
        return []
        
    # Deduplicate by content hash
    seen_hashes = set()
    deduped = []
    for doc in fused_results:
        content = doc["payload"].get("content", "")
        h = hashlib.md5(content.encode()).hexdigest()
        if h not in seen_hashes:
            seen_hashes.add(h)
            deduped.append(doc)
    fused_results = deduped
    
    fused_results = apply_supersession_flag(fused_results)
    pairs = [[query, doc["payload"]["content"]] for doc in fused_results]
    rerank_scores = reranker.predict(pairs)
    
    from retrieval.finbert_scorer import score_financial_relevance
    for i, doc in enumerate(fused_results):
        base_score = float(rerank_scores[i])
        if getattr(config, "ENABLE_FINBERT", False):
            finbert_score = score_financial_relevance(query, doc["payload"]["content"])
            doc["rerank_score"] = (1.0 - config.FINBERT_WEIGHT) * base_score + config.FINBERT_WEIGHT * finbert_score
        else:
            doc["rerank_score"] = base_score
            
    fused_results = sorted(fused_results, key=lambda x: x["rerank_score"], reverse=True)
    
    # Quality filter: remove chunks that are too short (TOC entries, headers, metadata)
    MIN_CONTENT_WORDS = 15
    quality_filtered = []
    for doc in fused_results:
        content = doc["payload"].get("content", "")
        word_count = len(content.split())
        if word_count >= MIN_CONTENT_WORDS:
            quality_filtered.append(doc)
    
    # If filtering removed everything, fall back to unfiltered results
    if not quality_filtered:
        quality_filtered = fused_results
    
    return quality_filtered[:config.TOP_K_RERANK]

def merge_deduplicate(res1: List[Dict], res2: List[Dict]) -> List[Dict]:
    seen = set()
    merged = []
    for r in res1 + res2:
        if r["id"] not in seen:
            seen.add(r["id"])
            merged.append(r)
    return merged

def rrf_three_way(dense_results: List[Dict], sparse_results: List[Dict], atomic_results: List[Dict], query: str = "", k: int = config.RRF_K) -> List[Dict]:
    scores = {}
    docs = {}
    
    for rank, res in enumerate(dense_results, 1):
        doc_id = res["id"]
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
        docs[doc_id] = res["payload"]
        
    for rank, res in enumerate(sparse_results, 1):
        doc_id = res["id"]
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
        docs[doc_id] = res["payload"]

    for rank, res in enumerate(atomic_results, 1):
        doc_id = res["id"]
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
        docs[doc_id] = res["payload"]
        
    q_lower = query.lower()
    for doc_id, payload in docs.items():
        domains = payload.get("domains", [])
        if domains:
            for d in domains:
                if d and d.lower() in q_lower:
                    scores[doc_id] *= 1.2
                    break
                
    sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [{"id": doc_id, "score": score, "payload": docs[doc_id]} for doc_id, score in sorted_docs]

def hybrid_search(query: str, domain_filter: str = None) -> List[Dict]:
    # HyDE: generate hypothetical doc for better dense retrieval
    if getattr(config, 'ENABLE_HYDE', False):
        hyde_doc = generate_hypothetical_document(query)
        hyde_dense_res = retrieve_dense(hyde_doc, domain_filter=domain_filter)
    else:
        hyde_dense_res = []

    dense_res = retrieve_dense(query, domain_filter=domain_filter)
    sparse_res = sparse_index.search(query, domain_filter=domain_filter)
    atomic_res = retrieve_atomic(query)
    
    all_dense = merge_deduplicate(dense_res, hyde_dense_res)
    fused_results = rrf_three_way(all_dense, sparse_res, atomic_res, query=query)
    
    return rerank_unioned_results(query, fused_results)
