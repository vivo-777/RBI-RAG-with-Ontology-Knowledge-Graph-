import pdfplumber
import fitz  # PyMuPDF
from typing import List, Dict, Any

def extract_text_and_tables(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Extracts text and tables page by page.
    """
    pages_data = []
    
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            tables = page.extract_tables()
            
            # Simple table formatting
            formatted_tables = []
            for table in tables:
                if not table:
                    continue
                # Convert table to markdown or CSV-like string
                md_table = "\n".join([" | ".join([str(cell) if cell else "" for cell in row]) for row in table if any(row)])
                if md_table:
                    formatted_tables.append(md_table)
                    
            pages_data.append({
                "page_number": page_num,
                "text": text or "",
                "tables": formatted_tables
            })
            
    return pages_data
