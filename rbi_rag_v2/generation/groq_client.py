import os
import time
import logging
from groq import Groq
import config

logger = logging.getLogger(__name__)

if not config.GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY is not set. Get your key from https://console.groq.com/keys "
        "and set it in rbi_rag_v2/.env"
    )

# Initialize the Groq client
client = Groq(api_key=config.GROQ_API_KEY)

try:
    from openai import OpenAI
    ollama_client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
except ImportError:
    ollama_client = None


def _is_rate_limit_error(error) -> bool:
    """Check if an error is a rate limit / quota error from Groq."""
    error_str = str(error).lower()
    return any(keyword in error_str for keyword in [
        "rate_limit", "429", "413", "too many requests",
        "tokens per minute", "requests per minute", "request too large"
    ])


def generate_groq_completion(prompt: str, stream: bool = False, model: str = None):
    """
    Reusable Groq client function with retry and fallback.

    Args:
        prompt: The user prompt to send.
        stream: If True, returns a generator that yields chunks.
        model: Override model ID. Defaults to config.GROQ_DEFAULT_MODEL.

    Returns:
        str or generator of str chunks.
    """
    model_to_use = model or config.GROQ_DEFAULT_MODEL

    messages = [{"role": "user", "content": prompt}]

    # Conservative token limits to stay within free tier TPM (12,000).
    # 2048 output tokens is plenty for regulatory answers (typically 200-500 tokens).
    max_tokens = 2048 if model_to_use == config.GROQ_DEFAULT_MODEL else 1024

    # Retry with exponential backoff for rate limits
    max_retries = 4
    for attempt in range(max_retries):
        try:
            if "qwen" in model_to_use.lower():
                if not ollama_client:
                    raise RuntimeError("openai python package is required for local Ollama fallback")
                completion = ollama_client.chat.completions.create(
                    model=model_to_use,
                    messages=messages,
                    temperature=0.1,
                    max_tokens=max_tokens,
                    top_p=1,
                    stream=stream,
                    stop=None
                )
            else:
                completion = client.chat.completions.create(
                    model=model_to_use,
                    messages=messages,
                    temperature=0.1,
                    max_completion_tokens=max_tokens,
                    top_p=1,
                    stream=stream,
                    stop=None
                )

            if stream:
                def stream_generator():
                    for chunk in completion:
                        content = chunk.choices[0].delta.content
                        if content:
                            yield content
                return stream_generator()
            else:
                return completion.choices[0].message.content

        except Exception as e:
            if _is_rate_limit_error(e) and attempt < max_retries - 1:
                wait_time = (2 ** (attempt + 1)) + 1  # 3, 5, 9 seconds
                logger.warning(f"Groq rate limit hit. Retrying in {wait_time}s (attempt {attempt + 1}/{max_retries})...")
                time.sleep(wait_time)
                continue

            # If it's not a rate limit or we've exhausted retries, try fallback model
            if model_to_use != config.GROQ_FAST_MODEL:
                logger.warning(f"Groq API call failed with {model_to_use}: {e}. Falling back to {config.GROQ_FAST_MODEL}")
                try:
                    time.sleep(2)  # Brief pause before fallback to avoid back-to-back rate limits
                    fallback_completion = client.chat.completions.create(
                        model=config.GROQ_FAST_MODEL,
                        messages=messages,
                        temperature=0.1,
                        max_completion_tokens=2048,
                        top_p=1,
                        stream=stream,
                        stop=None
                    )
                    if stream:
                        def fallback_stream():
                            for chunk in fallback_completion:
                                content = chunk.choices[0].delta.content
                                if content:
                                    yield content
                        return fallback_stream()
                    else:
                        return fallback_completion.choices[0].message.content
                except Exception as fallback_err:
                    logger.error(f"Groq fallback model also failed: {fallback_err}")
                    raise RuntimeError(
                        f"Groq primary ({model_to_use}) error: {e} | "
                        f"Fallback ({config.GROQ_FAST_MODEL}) error: {fallback_err}"
                    )
            else:
                logger.error(f"Groq API call failed: {e}")
                raise RuntimeError(f"Groq API error: {e}")
