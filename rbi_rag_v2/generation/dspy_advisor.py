"""
DSPy-based RBI Regulatory Advisor module.
Configured for Groq-only LLM backend.
"""

import re
import os
import logging
import time
from typing import List, Dict
import dspy
import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 1. DSPy Signature
# ---------------------------------------------------------------------------

class RBIRegulatoryAdvisor(dspy.Signature):
    """Answer an RBI regulatory question using ONLY the provided context blocks.
    ABSTENTION GUIDANCE: Prefer to answer the query using available context. If you only have partial information, state what you know and mark uncertain parts. Only if the context is completely irrelevant, state: 'The provided regulatory context does not contain sufficient information to answer this query definitively.'
    Cite every fact as [N] (Effective: YYYY-MM-DD) where N is the block ID and YYYY-MM-DD is the effective date from the Block ID header.
    Every single sentence in the answer must be fully grounded and supported by the context blocks."""

    context: str = dspy.InputField(desc="Retrieved, decontextualized circular chunks tagged with numeric block IDs")
    question: str = dspy.InputField(desc="The user's regulatory query")
    answer: str = dspy.OutputField(desc="Final answer as plain prose with inline [N] citations. No reasoning trace, no XML tags, no code fences.")


# ---------------------------------------------------------------------------
# 2. Sanitizer (hard backstop)
# ---------------------------------------------------------------------------

def sanitize_output(text: str) -> str:
    """Strip leaked tags, fences, and normalise citation formats."""
    # Strip <Thinking>...</Thinking> blocks (greedy, multiline)
    text = re.sub(r'<Thinking>.*?</Thinking>', '', text, flags=re.DOTALL | re.IGNORECASE)

    # Strip <FinalAnswer> / </FinalAnswer> tags
    text = re.sub(r'</?FinalAnswer>', '', text, flags=re.IGNORECASE)

    # Strip code fences
    text = re.sub(r'```(?:xml|json|python|text)?\s*', '', text)

    # Strip DSPy internal markers
    text = text.replace("[[ ## completed ## ]]", "")

    # Strip internal chunk-ref artifacts
    text = re.sub(r'\[?Block ID:\s*\d+\]?', '', text, flags=re.IGNORECASE)

    # Clean up spacing before punctuation
    text = re.sub(r'\s+([.,!?])', r'\1', text)
    text = re.sub(r' +', ' ', text)

    return text.strip()


# ---------------------------------------------------------------------------
# 3. Reward function for dspy.Refine
# ---------------------------------------------------------------------------

from generation.citation_verifier import verify_sentence, is_non_coverage_sentence

def advisor_reward(kwargs, prediction) -> float:
    answer = prediction.answer if hasattr(prediction, 'answer') else ""
    context = kwargs.get("context", "")
    score = 0.0

    blocks = context.split("\n\n")
    context_chunks = []
    for b in blocks:
        lines = b.strip().split("\n")
        if len(lines) > 1:
            context_chunks.append("\n".join(lines[1:]).strip())
        else:
            context_chunks.append(b.strip())

    sentences = [s for s in re.split(r'(?<=[.!?]) +', answer) if s.strip()]

    for sent in sentences:
        if not is_non_coverage_sentence(sent):
            if verify_sentence(sent, context_chunks) == "unsupported":
                return 0.0

    context_ids = set(re.findall(r'\[Block ID:\s*(\d+)', context))
    answer_ids = set(re.findall(r'\[(\d+)\]', answer))
    
    has_numbers = bool(re.search(r'\b\d+(?:\.\d+)?%?\b', answer))

    if not answer_ids:
        is_non_coverage = any(is_non_coverage_sentence(sent) for sent in sentences)
        if is_non_coverage or not has_numbers:
            score += 0.4
        else:
            score += 0.0
    elif answer_ids.issubset(context_ids):
        score += 0.4
    else:
        score += 0.1

    has_leaked = bool(re.search(r'<Thinking>|```|<FinalAnswer>', answer, re.IGNORECASE))
    if not has_leaked:
        score += 0.3
    else:
        score += 0.0

    ends_properly = bool(re.search(r'[.!?][\s"\']*$', answer.strip()))
    if ends_properly:
        score += 0.3
    else:
        score += 0.1

    return score


# ---------------------------------------------------------------------------
# 4. Module
# ---------------------------------------------------------------------------

class GroundedRBIAdvisor(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(RBIRegulatoryAdvisor)

    def forward(self, context: str, question: str, **kwargs):
        result = self.predict(context=context, question=question)
        result.answer = sanitize_output(result.answer)
        return result


# ---------------------------------------------------------------------------
# 5. Load / configure helper — Groq-only
# ---------------------------------------------------------------------------

_cached_advisors = {}

def load_advisor(model_name: str = None):
    """Configure DSPy with an Ollama model and return a Refine-wrapped advisor."""
    global _cached_advisors
    if model_name is None:
        model_name = config.GROQ_DEFAULT_MODEL

    # Configure DSPy with chosen Groq model
    try:
        # We explicitly use LiteLLM integration which DSPy supports out of the box.
        # It requires GROQ_API_KEY to be in the environment.
        os.environ["GROQ_API_KEY"] = config.GROQ_API_KEY
        # Return the configured LM object directly instead of globally configuring
        if "qwen" in model_name.lower():
            logger.info(f"Loaded LM for local Ollama/LMStudio model: ollama_chat/{model_name}")
            lm = dspy.LM(f"ollama_chat/{model_name}", api_base="http://localhost:11434")
        else:
            logger.info(f"Loaded LM for model: groq/{model_name}")
            lm = dspy.LM(f"groq/{model_name}")
    except Exception as e:
        logger.error(f"Failed to configure DSPy with Groq model {model_name}: {e}")
        return None, None

    try:
        dspy.configure_cache(enable_memory_cache=False, enable_disk_cache=False)
    except AttributeError:
        pass

    module = GroundedRBIAdvisor()

    compiled_path = os.path.join(os.path.dirname(__file__), "compiled_advisor.json")
    if os.path.exists(compiled_path):
        module.load(compiled_path)
        logger.info("Loaded compiled DSPy advisor from %s", compiled_path)
    else:
        logger.warning("No compiled advisor found at %s — using uncompiled module", compiled_path)

    _cached_advisors[model_name] = module
    return module, lm


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~1.3 tokens per whitespace-delimited word."""
    return int(len(text.split()) * 1.3)


def build_context_string(retrieved_chunks: List[Dict]) -> str:
    """Build context string with a hard token budget to stay within Groq free tier limits.
    
    Chunks are added in rerank order (highest relevance first). If a chunk
    would exceed the budget, it is truncated to fit. Remaining chunks are dropped.
    """
    budget = config.MAX_CONTEXT_TOKENS
    used = 0
    parts = []

    for idx, chunk in enumerate(retrieved_chunks, start=1):
        payload = chunk["payload"]
        filename = payload.get("filename", "Unknown")
        effective_from = payload.get("effective_from", "Unknown")
        content = payload.get("content", "")

        header = f"[Block ID: {idx}, Source: {filename}, Effective Date: {effective_from}]"
        header_tokens = _estimate_tokens(header)
        content_tokens = _estimate_tokens(content)
        block_tokens = header_tokens + content_tokens

        if used + block_tokens <= budget:
            # Fits entirely
            parts.append(f"{header}\n{content}")
            used += block_tokens
        elif used + header_tokens < budget:
            # Partially fits — truncate content to fill remaining budget
            remaining_tokens = budget - used - header_tokens
            remaining_words = int(remaining_tokens / 1.3)
            if remaining_words > 20:  # Only include if meaningful content remains
                truncated = " ".join(content.split()[:remaining_words]) + " [...]"
                parts.append(f"{header}\n{truncated}")
                used = budget  # Budget exhausted
                logger.info(f"Truncated chunk {idx} to ~{remaining_words} words to fit token budget.")
            break
        else:
            # No room left at all
            logger.info(f"Dropped chunks {idx}–{len(retrieved_chunks)} due to token budget ({used}/{budget} used).")
            break

    logger.info(f"Context built: {len(parts)} chunks, ~{used} estimated tokens (budget: {budget}).")
    return "\n\n".join(parts)


def _run_pipeline(query: str, model_name: str, hybrid_search, QueryClassifier, QueryDecomposer, parallel_subquery_retrieve):
    """Run the full classify → retrieve → generate pipeline with a specific Groq model."""
    advisor, lm = load_advisor(model_name)
    if not advisor:
        raise RuntimeError(f"Failed to load advisor for {model_name}")

    # 1. Pre-routing Classifier
    classifier = QueryClassifier()
    with dspy.context(lm=lm):
        category = classifier(query=query)

    # 2. Always retrieve first, even for OUT_OF_CONTEXT — classifier can be wrong
    if category == "COMPOUND":
        decomposer = QueryDecomposer()
        with dspy.context(lm=lm):
            sub_queries = decomposer(query=query)
        retrieved = parallel_subquery_retrieve(sub_queries, original_query=query)
    else:
        retrieved = hybrid_search(query)

    # 2b. Retrieval-gated OUT_OF_CONTEXT override
    if category == "OUT_OF_CONTEXT":
        max_rerank = max((c.get("rerank_score", 0.0) for c in retrieved), default=0.0)
        if max_rerank < 0.5:
            # Low rerank scores confirm the classifier — truly out of context
            logger.info(f"OUT_OF_CONTEXT confirmed by reranker (max_score={max_rerank:.4f})")
            return {
                "answer": "The provided query is out of context and does not relate to RBI regulatory circulars.",
                "retrieved": [],
                "model": model_name,
                "is_oot": True
            }
        else:
            # High rerank scores override the classifier — the query IS relevant
            logger.warning(f"OUT_OF_CONTEXT overridden by reranker (max_score={max_rerank:.4f}). Proceeding with generation.")
            category = "SIMPLE"  # Treat as a normal query

    if not retrieved:
        return {
            "answer": "The provided regulatory context does not contain sufficient information to answer this query definitively.",
            "retrieved": [],
            "model": model_name,
        }

    # 3. Confidence-gated escalation
    actual_model = model_name
    max_rerank = max(c.get("rerank_score", 0.0) for c in retrieved)
    if max_rerank < config.RERANK_CONFIDENCE_THRESHOLD:
        actual_model = config.GROQ_ESCALATION_MODEL
        logger.info(f"Reranker top-score {max_rerank:.4f} below threshold. Escalating to {actual_model}.")
    else:
        logger.info(f"Reranker top-score {max_rerank:.4f} at or above threshold. Using {actual_model}.")

    # 4. Format Context
    context_str = build_context_string(retrieved)

    # 5. Generate Answer (with fallback to escalation model)
    answer = None
    gen_models = [actual_model]
    if config.GROQ_ESCALATION_MODEL not in gen_models:
        gen_models.append(config.GROQ_ESCALATION_MODEL)

    votes = []
    for attempt_model in gen_models:
        try:
            logger.info(f"Attempting generation with model: {attempt_model}")
            attempt_advisor, attempt_lm = load_advisor(attempt_model)
            if not attempt_advisor:
                continue
            # Main generation (temp=0) using thread-safe context
            with dspy.context(lm=attempt_lm):
                result = attempt_advisor(context=context_str, question=query)
                
            if result is not None and hasattr(result, 'answer') and result.answer:
                answer = sanitize_output(result.answer)

                # Self-consistency votes removed to prevent hitting Groq 12000 TPM limit
                votes = []

                actual_model = attempt_model
                break
            else:
                logger.warning(f"Model {attempt_model} returned None/empty result, trying next...")
        except Exception as gen_err:
            if "RateLimitError" in str(gen_err):
                raise  # Bubble up rate limits so the script can wait
            logger.warning(f"Model {attempt_model} generation failed: {gen_err}")

    if answer is None:
        raise RuntimeError("All generation models failed")

    return {
        "answer": answer,
        "votes": votes,
        "retrieved": retrieved,
        "model": actual_model,
    }



def generate_dspy_answer(query: str, retrieved_chunks: List[Dict] = None) -> dict:
    start = time.time()

    from retrieval.query_decomposer import QueryClassifier, QueryDecomposer, parallel_subquery_retrieve, expand_query
    from retrieval.fusion import hybrid_search
    
    query = expand_query(query)

    # Groq-only pipeline: try default model, then escalation model
    pipeline_models = [config.GROQ_DEFAULT_MODEL]
    if config.GROQ_ESCALATION_MODEL != config.GROQ_DEFAULT_MODEL:
        pipeline_models.append(config.GROQ_ESCALATION_MODEL)

    for pipeline_model in pipeline_models:
        try:
            logger.info(f"Running full pipeline with Groq model: {pipeline_model}")
            result = _run_pipeline(
                query, pipeline_model, hybrid_search,
                QueryClassifier, QueryDecomposer, parallel_subquery_retrieve
            )
            
            if result.get("is_oot"):
                return {
                    "answer": result["answer"],
                    "retries": 0,
                    "latency": time.time() - start,
                    "model": result["model"]
                }

            return {
                "answer": result["answer"],
                "votes": result.get("votes", []),
                "retries": 0,
                "latency": time.time() - start,
                "retrieved_chunks": result["retrieved"],
                "model": result["model"]
            }

        except ValueError as ve:
            return {
                "answer": f"The requested regulatory domain is not yet ingested in our compliance database: {str(ve)}",
                "retries": 0,
                "latency": time.time() - start,
                "retrieved_chunks": [],
                "model": pipeline_model
            }
        except Exception as e:
            if "RateLimitError" in str(e):
                raise  # Bubble up rate limit out of generate_dspy_answer
            logger.warning(f"Full pipeline with {pipeline_model} failed: {e}. Trying fallback...")

    # All models exhausted
    logger.error("All Groq pipeline models failed for query: %s", query[:80])
    return {
        "answer": "The provided regulatory context does not contain sufficient information to answer this query definitively.",
        "retries": 0,
        "latency": time.time() - start,
        "retrieved_chunks": [],
        "model": "none"
    }
