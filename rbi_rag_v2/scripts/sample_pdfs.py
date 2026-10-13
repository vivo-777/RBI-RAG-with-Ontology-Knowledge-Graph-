import os, pdfplumber

d='../circulars'
files=[f for f in os.listdir(d) if f.endswith('.pdf') or f.endswith('.PDF')]

with open('pdf_samples.txt', 'w', encoding='utf-8') as out:
    for f in files:
        try:
            with pdfplumber.open(os.path.join(d,f)) as pdf:
                t = (pdf.pages[0].extract_text() or '')[:500]
                out.write(f"--- {f} ---\n{t.replace(chr(10), ' ')}\n\n")
        except Exception as e:
            out.write(f"--- {f} ---\nERROR: {e}\n\n")
