import os
import fitz  # PyMuPDF

script_dir = os.path.dirname(os.path.abspath(__file__))
pdf_dir = os.path.abspath(os.path.join(script_dir, "..", "..", "circulars"))
query = "perpetual debt instrument"

results = []

for filename in os.listdir(pdf_dir):
    if filename.lower().endswith('.pdf'):
        filepath = os.path.join(pdf_dir, filename)
        try:
            doc = fitz.open(filepath)
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text().lower()
                if query in text:
                    results.append((filename, page_num + 1))
        except Exception as e:
            pass

if results:
    print(f"Found '{query}' in the following circulars:")
    for res in results:
        print(f" - {res[0]} (Page {res[1]})")
else:
    print(f"'{query}' not found in any circulars.")
