import re
from typing import List, Dict, Any

def parse_structure(text: str) -> List[Dict[str, str]]:
    """
    Structure-aware chapter/section regex parser.
    Splits text by common section headers (e.g. '1. Introduction', 'Chapter II').
    """
    # Expanded regex for matching Section headers
    pattern = r'^(?:Chapter\s+(?:\w+|[IVXLCDM]+)|[0-9]+(?:\.[0-9]+)*\.?|\([a-zA-Z0-9ivxlcdm]+\)|[a-zA-Z]\)|[IVXLCDM]+\.|Annexure\s+\w+|Appendix\s+\w+|Part\s+[A-Z0-9IVXLCDM]+|Section\s+\d+|\*\*.*?\*\*|Table\s+[IVXLCDM0-9]+|(?:Introduction|Background|Scope|Eligibility|Guidelines|Instructions|Objectives|Features|Scheme)\b)\s*.*$'
    
    parts = re.split(f'(?im)({pattern})', text)
    
    sections = []
    current_header = "General"
    
    if parts and not re.match(pattern, parts[0], flags=re.MULTILINE):
        if parts[0].strip():
            sections.append({"header": current_header, "content": parts[0].strip()})
        parts = parts[1:]
        
    for i in range(0, len(parts), 2):
        if i < len(parts):
            current_header = parts[i].strip()
        if i + 1 < len(parts):
            content = parts[i+1].strip()
            if content:
                sections.append({"header": current_header, "content": content})
                
    if not sections and text.strip():
        sections.append({"header": current_header, "content": text.strip()})
        
    return sections
