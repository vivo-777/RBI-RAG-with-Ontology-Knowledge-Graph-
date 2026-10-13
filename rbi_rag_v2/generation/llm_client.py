"""
LLM client — Groq streaming wrapper.
Uses the Groq client for streaming generation.
"""
from typing import Iterator
from generation.groq_client import generate_groq_completion


def generate_stream(prompt: str) -> Iterator[str]:
    """
    Streams output from Groq API.
    """
    return generate_groq_completion(prompt, stream=True)
