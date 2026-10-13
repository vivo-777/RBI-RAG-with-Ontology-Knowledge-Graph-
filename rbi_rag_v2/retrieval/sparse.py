from rank_bm25 import BM25Okapi
import config
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

class SparseIndex:
    def __init__(self):
        self.corpus = []
        self.payloads = []
        self.bm25 = None
        self._stemmer = PorterStemmer()
        self._stops = set(stopwords.words('english'))
        
    def _tokenize(self, text: str) -> list:
        """Lowercase, remove punctuation, stem, drop stopwords."""
        tokens = word_tokenize(text.lower())
        return [self._stemmer.stem(t) for t in tokens
                if t.isalnum() and t not in self._stops]
        
    def add(self, document: str, payload: dict):
        self.corpus.append(document)
        self.payloads.append(payload)
        
    def build(self):
        if self.corpus:
            tokenized_corpus = [self._tokenize(doc) for doc in self.corpus]
            self.bm25 = BM25Okapi(tokenized_corpus)
            
    def save(self, filepath: str):
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump({"corpus": self.corpus, "payloads": self.payloads, "bm25": self.bm25}, f)
            
    def load(self, filepath: str):
        import pickle
        import os
        import hashlib
        if os.path.exists(filepath):
            # Basic integrity check: verify file size is reasonable
            file_size = os.path.getsize(filepath)
            if file_size > 100_000_000:  # 100MB sanity check
                raise ValueError(f"BM25 index file suspiciously large: {file_size} bytes")
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                self.corpus = data["corpus"]
                self.payloads = data["payloads"]
                self.bm25 = data["bm25"]
            
    def search(self, query: str, top_k: int = config.TOP_K_SPARSE, domain_filter: str = None):
        if not self.bm25:
            return []
            
        tokenized_query = self._tokenize(query)
        doc_scores = self.bm25.get_scores(tokenized_query)
        
        # Sort and filter
        scored_docs = []
        for i, score in enumerate(doc_scores):
            if score > 0:
                payload = self.payloads[i]
                if domain_filter and domain_filter not in payload.get("domains", []):
                    continue
                scored_docs.append({
                    "id": payload.get("chunk_id", str(i)),
                    "score": score,
                    "payload": payload
                })
                
        scored_docs = sorted(scored_docs, key=lambda x: x["score"], reverse=True)
        return scored_docs[:top_k]

# Global sparse index for demo purposes
sparse_index = SparseIndex()

import os
bm25_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "bm25_data.pkl"))
sparse_index.load(bm25_path)
