"""
Hypothetical Document Embedding (HyDE) for bridging vocabulary gap
between user queries and regulatory text.
"""
import logging
from generation.groq_client import generate_groq_completion
import config

logger = logging.getLogger(__name__)

def generate_hypothetical_document(query: str) -> str:
    """Ask the LLM to generate a hypothetical regulatory paragraph 
    that would answer the query. This is then embedded for retrieval."""
    prompt = f"""You are an RBI regulatory expert. Write a brief, factual paragraph 
(2-3 sentences) that would directly answer this regulatory query. 
Use specific regulatory terminology, circular numbers, and numeric thresholds 
as they would appear in an actual RBI circular.

Query: {query}

Hypothetical regulatory paragraph:"""
    
    try:
        result = generate_groq_completion(
            prompt, stream=False, model=config.GROQ_FAST_MODEL
        )
        return result.strip()
    except Exception as e:
        logger.warning(f"HyDE generation failed: {e}, falling back to raw query")
        return query
