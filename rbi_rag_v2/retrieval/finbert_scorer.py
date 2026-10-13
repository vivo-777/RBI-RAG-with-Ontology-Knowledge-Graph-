import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
import config
import logging

logger = logging.getLogger(__name__)
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = None
model = None

if getattr(config, "ENABLE_FINBERT", False):
    try:
        tokenizer = AutoTokenizer.from_pretrained(config.FINBERT_MODEL)
        model = AutoModel.from_pretrained(config.FINBERT_MODEL).to(device)
        model.eval()
    except Exception as e:
        logger.error(f"Failed to load FinBERT model: {e}")

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def score_financial_relevance(query: str, chunk: str) -> float:
    if not model or not tokenizer:
        return 0.0
        
    try:
        # Tokenize sentences
        encoded_input = tokenizer([query, chunk], padding=True, truncation=True, max_length=512, return_tensors='pt').to(device)
        
        # Compute token embeddings
        with torch.no_grad():
            model_output = model(**encoded_input)
            
        # Perform pooling
        sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
        
        # Normalize embeddings
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        
        # Compute cosine similarity
        query_emb = sentence_embeddings[0]
        chunk_emb = sentence_embeddings[1]
        
        cosine_sim = torch.dot(query_emb, chunk_emb).item()
        
        # Map from [-1, 1] to roughly [0, 1] to match cross-encoder scale
        return max(0.0, cosine_sim)
    except Exception as e:
        logger.error(f"FinBERT scoring failed: {e}")
        return 0.0
