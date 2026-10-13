import os
from dotenv import load_dotenv

# Load .env BEFORE any os.getenv() calls
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"), override=True)

# Database & Retrieval
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
QDRANT_COLLECTION = "rbi_circulars"

# Models
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

# Groq API Configuration (Groq-only pipeline)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_DEFAULT_MODEL = os.getenv("GROQ_DEFAULT_MODEL", "llama-3.3-70b-versatile")
GROQ_ESCALATION_MODEL = os.getenv("GROQ_ESCALATION_MODEL", "llama-3.3-70b-versatile")
GROQ_FAST_MODEL = os.getenv("GROQ_FAST_MODEL", "llama-3.1-8b-instant")

# HyDE (Hypothetical Document Embedding)
ENABLE_HYDE = os.getenv("ENABLE_HYDE", "true").lower() == "true"

RERANK_CONFIDENCE_THRESHOLD = float(os.getenv("RERANK_CONFIDENCE_THRESHOLD", "0.3"))
RERANK_COSINE_SIMILARITY_THRESHOLD = float(os.getenv("COSINE_SIMILARITY_THRESHOLD", "0.6"))

# Fusion & Reranking Settings
RRF_K = 60
TOP_K_DENSE = 40
TOP_K_SPARSE = 40
TOP_K_RERANK = 20

# Groq Free Tier Token Budget
# Free tier allows 12,000 TPM. Reserve ~2,048 for output + ~500 for system prompt/signature.
# That leaves ~9,400 tokens for context. At ~1.3 tokens/word, that's ~7,200 words max.
MAX_CONTEXT_TOKENS = int(os.getenv("MAX_CONTEXT_TOKENS", "11000"))

# Groundedness Scorer Thresholds
JACCARD_OVERLAP_THRESHOLD = 0.30
COSINE_SIMILARITY_THRESHOLD = 0.55

# API Settings
API_HOST = "127.0.0.1"
API_PORT = 8543

# FinBERT Settings
FINBERT_MODEL = "ProsusAI/finbert"
FINBERT_WEIGHT = 0.3
ENABLE_FINBERT = True
