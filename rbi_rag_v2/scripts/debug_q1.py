"""Debug: check what pdfplumber extracts from BASEL III CAPITAL FRAMEWORK.pdf"""
import pdfplumber
import os

pdf_path = os.path.join("..", "circulars", "BASEL III CAPITAL FRAMEWORK.pdf")
pdf_path = os.path.abspath(pdf_path)

with open("debug_q1_out.txt", "w", encoding="utf-8") as f:
    with pdfplumber.open(pdf_path) as pdf:
        f.write(f"Total pages: {len(pdf.pages)}\n")
        f.write(f"Metadata: {pdf.metadata}\n\n")
        for i, page in enumerate(pdf.pages, 1):
            text = page.extract_text() or ""
            f.write(f"\n=== Page {i} ({len(text)} chars) ===\n")
            f.write(text[:1000] if text else "(EMPTY - no text extracted)")
            f.write("\n...\n")
        
        # Check total extractable text
        total = sum(len((p.extract_text() or "")) for p in pdf.pages)
        f.write(f"\n=== TOTAL TEXT: {total} chars across {len(pdf.pages)} pages ===\n")
        f.write(f"Would trigger OCR check (< 200 chars)? {total < 200}\n")
