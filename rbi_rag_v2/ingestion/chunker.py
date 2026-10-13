from typing import List, Dict, Any

def recursive_chunk_text(text: str, max_words: int = 400, overlap: int = 150) -> List[str]:
    """
    Recursive sliding-window chunker with sentence-boundary awareness.
    """
    words = text.split()
    chunks = []
    
    if len(words) <= max_words:
        return [text]
        
    start = 0
    while start < len(words):
        end = start + max_words
        
        # Try to find a period near the end to split cleanly
        if end < len(words):
            for i in range(end - 1, max(start + max_words - 40, start), -1):
                if words[i].endswith('.'):
                    end = i + 1
                    break
                    
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start = end - overlap
        if start >= end:
            start = end - 20 # Safeguard against zero progress
        
    return chunks

def chunk_section(section: Dict[str, str], max_words: int = 200) -> List[Dict[str, str]]:
    """
    Chunks a section while preserving its header.
    """
    text_chunks = recursive_chunk_text(section["content"], max_words)
    return [{"header": section["header"], "content": chunk} for chunk in text_chunks]
