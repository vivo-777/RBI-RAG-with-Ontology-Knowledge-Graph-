import re
from typing import List, Dict, Any
from sentence_transformers import util
import config
from retrieval.dense import embed_model
from generation.groq_client import generate_groq_completion
import logging
import functools

@functools.lru_cache(maxsize=1)
def _cached_all_facts():
    """Cache all facts in memory — cleared on process restart."""
    from ontology.facts_store import get_all_facts
    return get_all_facts()

def check_a_facts_db(sentence: str) -> bool:
    numbers = set(re.findall(r'\b\d+(?:\.\d+)?\b', sentence))
    if not numbers:
        return True # Not a numeric claim, passes Check A by default
    
    # Simple heuristic: try to find a fact that matches
    # In a real app we'd extract the metric name using an LLM. 
    # Here we just fetch all facts and check if the number exists in any value.
    # To be somewhat strict, we fetch all facts from DB (or search by some keywords)
    # We will just grab all facts for simplicity since the DB is small for testing.
    # We will just grab all facts for simplicity since the DB is small for testing.
    try:
        all_facts = _cached_all_facts()
        for num in numbers:
            found = False
            for fact in all_facts:
                if str(num) in str(fact.get("value", "")) or str(num) in str(fact.get("raw_context", "")):
                    found = True
                    break
            if not found:
                return False
        return True
    except Exception as e:
        logging.error(f"Facts DB Check A failed: {e}")
        return False

def check_b_text_grounding(sentence: str, context_chunks: List[str]) -> float:
    sent_emb = embed_model.encode(sentence)
    best_cosine = 0.0
    for chunk in context_chunks:
        chunk_emb = embed_model.encode(chunk)
        cosine = util.cos_sim(sent_emb, chunk_emb).item()
        if cosine > best_cosine:
            best_cosine = cosine
    return best_cosine

def adversarial_critique(claim: str, context: str) -> bool:
    prompt = f"""You are a strict regulatory auditor. Given the evidence, argue why the following claim might be WRONG.
If the claim is perfectly supported by the evidence and cannot be argued against, output "SUPPORTED".
Otherwise, explain the flaw.
Evidence: {context}
Claim: {claim}"""
    try:
        # Force the evaluator to use the extremely cheap 8B model instead of burning the 70B daily limit
        critique = generate_groq_completion(prompt, stream=False, model=config.GROQ_FAST_MODEL)
        if "SUPPORTED" in critique.upper()[:20]:
            return True
        logging.info(f"Adversarial critique flagged claim: {claim}\nCritique: {critique}")
        return False
    except Exception as e:
        logging.error(f"Critique failed: {e}")
        return True # default to pass if API fails

def verify_sentence(sentence: str, context_chunks: List[str]) -> str:
    """Returns 'supported', 'partially_supported', or 'unsupported'."""
    has_numbers = bool(re.findall(r'\b\d+(?:\.\d+)?\b', sentence))
    
    pass_a = check_a_facts_db(sentence) if has_numbers else True
    best_cosine = check_b_text_grounding(sentence, context_chunks)
    
    pass_b = best_cosine >= config.COSINE_SIMILARITY_THRESHOLD
    partial_match = best_cosine >= (config.COSINE_SIMILARITY_THRESHOLD - 0.10)
    
    if pass_a and pass_b:
        # High risk claim -> adversarial critique
        if has_numbers:
            context_text = " ".join(context_chunks)
            if not adversarial_critique(sentence, context_text):
                return "partially_supported"
        return "supported"
    
    if pass_a or pass_b:
        return "partially_supported"
        
    if partial_match:
        # Check keyword overlap as fallback
        words_sent = set(re.findall(r'\w+', sentence.lower()))
        words_ctx = set(re.findall(r'\w+', " ".join(context_chunks).lower()))
        overlap = len(words_sent.intersection(words_ctx)) / max(1, len(words_sent))
        if overlap >= 0.30:
            return "partially_supported"
            
    return "unsupported"

def is_non_coverage_sentence(sentence: str) -> bool:
    # A helper for DSPy reward function to ignore abstention strings
    return "does not contain sufficient information" in sentence.lower()

def verify_response(response: str, context_chunks: List[str]) -> Dict[str, Any]:
    sentences = [s for s in re.split(r'(?<=[.!?]) +', response) if s.strip()]
    
    verified_sentences = []
    claim_labels = []
    supported_count = 0
    
    for sent in sentences:
        label = verify_sentence(sent, context_chunks)
        claim_labels.append({"claim": sent, "label": label})
        if label == "supported":
            verified_sentences.append(sent)
            supported_count += 1
        elif label == "partially_supported":
            verified_sentences.append(sent + " [⚠️ Requires Analyst Review]")
            supported_count += 0.5
            
    total = len(sentences)
    accuracy = (supported_count / total) if total > 0 else 1.0
            
    if not verified_sentences:
        return {
            "response": "The provided regulatory context does not contain sufficient information to answer this query definitively.",
            "accuracy": 0.0,
            "claim_labels": claim_labels
        }
        
    return {
        "response": " ".join(verified_sentences),
        "accuracy": accuracy,
        "claim_labels": claim_labels
    }

