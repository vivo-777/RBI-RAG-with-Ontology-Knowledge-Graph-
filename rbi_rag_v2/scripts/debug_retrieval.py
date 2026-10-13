"""Diagnose ALL PDFs: compare actual page count & file size vs indexed chunk count."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
os.environ["PYTHONIOENCODING"] = "utf-8"

import fitz  # PyMuPDF for page count
from collections import Counter

# --- 1. Get chunk counts from Qdrant ---
import config
from qdrant_client import QdrantClient

db_path = os.getenv("QDRANT_DATA_PATH", os.path.abspath(os.path.join(os.path.dirname(__file__), "qdrant_data_v2")))
client = QdrantClient(path=db_path)

all_points = client.scroll(config.QDRANT_COLLECTION, limit=10000, with_payload=True)[0]
chunk_counts = Counter(p.payload.get("filename", "unknown") for p in all_points)

# Also get chunk sizes (word counts) per file
chunk_sizes_by_file = {}
for p in all_points:
    fn = p.payload.get("filename", "unknown")
    content = p.payload.get("content", "")
    word_count = len(content.split())
    if fn not in chunk_sizes_by_file:
        chunk_sizes_by_file[fn] = []
    chunk_sizes_by_file[fn].append(word_count)

# --- 2. Get actual PDF stats ---
circulars_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "circulars"))
pdf_stats = {}
for f in os.listdir(circulars_dir):
    if f.lower().endswith(".pdf"):
        path = os.path.join(circulars_dir, f)
        size_kb = os.path.getsize(path) / 1024
        try:
            doc = fitz.open(path)
            pages = len(doc)
            # Count total words extracted
            total_words = 0
            for page in doc:
                total_words += len(page.get_text().split())
            doc.close()
        except:
            pages = -1
            total_words = 0
        pdf_stats[f] = {"pages": pages, "size_kb": size_kb, "extracted_words": total_words}

# --- 3. Print comparison table ---
print("=" * 130)
print(f"{'PDF Filename':<75} {'Pages':>5} {'Size(KB)':>9} {'Words':>7} {'Chunks':>7} {'Words/Chunk':>11} {'Ratio':>7}")
print("=" * 130)

rows = []
for fn, stats in sorted(pdf_stats.items()):
    chunks = chunk_counts.get(fn, 0)
    pages = stats["pages"]
    size_kb = stats["size_kb"]
    words = stats["extracted_words"]
    
    if chunks > 0:
        sizes = chunk_sizes_by_file.get(fn, [])
        avg_words = sum(sizes) / len(sizes) if sizes else 0
    else:
        avg_words = 0
    
    # Ratio: expected chunks vs actual (rough heuristic: ~1 chunk per 200 words)
    expected_chunks = max(1, words // 200)
    ratio = chunks / expected_chunks if expected_chunks > 0 else 0
    
    flag = ""
    if chunks == 0:
        flag = " *** NOT INDEXED ***"
    elif ratio < 0.3:
        flag = " *** CRITICALLY LOW ***"
    elif ratio < 0.6:
        flag = " ** LOW **"
    elif avg_words > 400:
        flag = " * OVERSIZED CHUNKS *"
    
    rows.append((fn, pages, size_kb, words, chunks, avg_words, ratio, flag))

# Sort by ratio (worst first)
rows.sort(key=lambda r: r[6])

for fn, pages, size_kb, words, chunks, avg_words, ratio, flag in rows:
    print(f"{fn:<75} {pages:>5} {size_kb:>9.0f} {words:>7} {chunks:>7} {avg_words:>11.0f} {ratio:>7.2f}{flag}")

print("\n" + "=" * 130)
print("\nLEGEND:")
print("  Ratio = actual_chunks / expected_chunks (expected = total_words / 200)")
print("  Ratio < 0.3 = CRITICALLY LOW (severe under-chunking)")
print("  Ratio < 0.6 = LOW (moderate under-chunking)")
print("  Words/Chunk > 400 = OVERSIZED (chunks too big for good embeddings)")

# --- 4. Detailed analysis of problematic PDFs ---
print("\n\n" + "=" * 130)
print("DETAILED ANALYSIS OF PROBLEMATIC PDFs")
print("=" * 130)

for fn, pages, size_kb, words, chunks, avg_words, ratio, flag in rows:
    if flag and chunks > 0:
        print(f"\n--- {fn} ---")
        print(f"  Pages: {pages}, Extracted words: {words}, Indexed chunks: {chunks}")
        sizes = chunk_sizes_by_file.get(fn, [])
        if sizes:
            print(f"  Chunk word counts: min={min(sizes)}, max={max(sizes)}, avg={sum(sizes)/len(sizes):.0f}, median={sorted(sizes)[len(sizes)//2]}")
        
        # Check if text extraction is actually working
        path = os.path.join(circulars_dir, fn)
        try:
            doc = fitz.open(path)
            for pg_idx in range(min(3, len(doc))):
                pg_text = doc[pg_idx].get_text()
                word_count = len(pg_text.split())
                preview = pg_text[:200].replace('\n', ' ').strip()
                print(f"  Page {pg_idx+1} ({word_count} words): {preview}...")
            doc.close()
        except Exception as e:
            print(f"  Error reading PDF: {e}")

client.close()
