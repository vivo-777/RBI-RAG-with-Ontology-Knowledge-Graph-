import hashlib
import os
import re
from typing import Set
import pdfplumber

def get_file_sha256(filepath: str) -> str:
    """Calculate SHA-256 of raw bytes of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            sha256.update(chunk)
    return sha256.hexdigest()

def normalize_text(text: str) -> str:
    """Normalize text by stripping whitespace, converting to lowercase, and removing common headers/footers."""
    # Lowercase
    text = text.lower()
    # Remove common headers/footers patterns (e.g., page numbers, dates)
    text = re.sub(r'\bpage \d+ of \d+\b', '', text)
    text = re.sub(r'\bpage \d+\b', '', text)
    # Strip all whitespaces/non-alphanumeric chars to compare semantic content
    text = re.sub(r'[^a-z0-9]', '', text)
    return text

def get_pdf_text_hash(filepath: str) -> str:
    """Extract text, normalize, and calculate SHA-256 hash of the normalized text."""
    full_text = ""
    try:
        with pdfplumber.open(filepath) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    full_text += text
    except Exception as e:
        print(f"Error reading PDF text for duplicate check: {e}")
        
    normalized = normalize_text(full_text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()
