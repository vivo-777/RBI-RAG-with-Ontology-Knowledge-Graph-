import pdfplumber
import fitz  # PyMuPDF
import easyocr
import numpy as np
from typing import List, Dict, Any
import os

reader = None

def get_ocr_reader():
    global reader
    if reader is None:
        reader = easyocr.Reader(['en'])
    return reader

def extract_text_with_ocr(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Extracts text and tables page by page. Uses EasyOCR on rendered pages.
    """
    pages_data = []
    doc = fitz.open(pdf_path)
    
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, (plumber_page, fitz_page) in enumerate(zip(pdf.pages, doc), start=1):
            # Render page to image (dpi=150 is good enough for OCR usually)
            pix = fitz_page.get_pixmap(dpi=150)
            
            # EasyOCR expects BGR or RGB numpy array or filepath
            # For fitz, pix.samples is bytes. If it has alpha (RGBA), we should drop it, but typically PDF pages are RGB.
            if pix.alpha:
                # Convert RGBA to RGB
                img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape((pix.height, pix.width, 4))[:, :, :3]
            else:
                if pix.n == 1:
                    # Grayscale
                    img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape((pix.height, pix.width))
                else:
                    # RGB
                    img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape((pix.height, pix.width, 3))
            
            ocr_reader = get_ocr_reader()
            results = ocr_reader.readtext(img_array, detail=0, paragraph=True)
            text = "\n\n".join(results)
            
            tables = plumber_page.extract_tables()
            formatted_tables = []
            for table in tables:
                if not table:
                    continue
                md_table = "\n".join([" | ".join([str(cell) if cell else "" for cell in row]) for row in table if any(row)])
                if md_table:
                    formatted_tables.append(md_table)
                    
            pages_data.append({
                "page_number": page_num,
                "text": text or "",
                "tables": formatted_tables
            })
            
    return pages_data
