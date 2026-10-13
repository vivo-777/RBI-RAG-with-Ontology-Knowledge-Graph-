"""
Re-run ONLY the 12 failing queries with the fixed pipeline,
then update both tailscale_40_results.json and tailscale_40_results.docx.
"""
import json, os, sys, time
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
config.GROQ_DEFAULT_MODEL = "qwen2.5:7b-instruct"
config.GROQ_ESCALATION_MODEL = "qwen2.5:7b-instruct"
config.MAX_CONTEXT_TOKENS = 11000

from generation.dspy_advisor import generate_dspy_answer

FAILING_IDS = [5, 8, 25, 28, 31, 33, 34, 35, 37, 38, 39, 40]

# Load original results
results_path = os.path.join(os.path.dirname(__file__), "tailscale_40_results.json")
# We need the ORIGINAL results file - restore it from the first successful run
# The user still has it open, so let's use the backup approach
queries_path = os.path.join(os.path.dirname(__file__), "eval", "queries.json")
with open(queries_path, "r") as f:
    all_queries = json.load(f)[:40]

# Load the existing results that the user has (from the old run)
old_results_path = os.path.join(os.path.dirname(__file__), "tailscale_40_results_backup.json")
if os.path.exists(results_path):
    with open(results_path, "r") as f:
        results = json.load(f)
elif os.path.exists(old_results_path):
    with open(old_results_path, "r") as f:
        results = json.load(f)
else:
    print("ERROR: No results file found to update!")
    sys.exit(1)

print(f"Loaded {len(results)} existing results")
print(f"Re-running {len(FAILING_IDS)} failing queries with fixed pipeline...")

for qid in FAILING_IDS:
    q = all_queries[qid - 1]
    query_text = q["question"]
    print(f"\n[{qid}/40] Re-evaluating: {query_text}")
    start = time.time()
    
    try:
        dspy_result = generate_dspy_answer(query_text)
        answer = dspy_result.get("answer", "")
        latency = time.time() - start
        
        print(f"  --> Answer: {answer[:150]}...")
        print(f"  --> Latency: {latency:.2f}s")
        
        # Update the result in-place (0-indexed)
        idx = qid - 1
        if idx < len(results):
            results[idx]["answer"] = answer
            results[idx]["latency"] = latency
        
    except Exception as e:
        print(f"  ERROR on query {qid}: {e}")

# Save updated JSON
with open(results_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4, ensure_ascii=False)
print(f"\nUpdated JSON saved to {results_path}")

# Regenerate DOCX
try:
    from docx import Document
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "python-docx"])
    from docx import Document

docx_path = os.path.join(os.path.dirname(__file__), "tailscale_40_results.docx")
doc = Document()
doc.add_heading("Evaluation Results (Tailscale - Qwen 2.5 7B) [UPDATED]", 0)

for i, res in enumerate(results):
    doc.add_heading(f"Query {i+1}:", level=2)
    doc.add_paragraph(res.get("query", ""))
    doc.add_heading("Answer:", level=3)
    doc.add_paragraph(res.get("answer", ""))
    latency_p = doc.add_paragraph()
    latency_p.add_run(f"Latency: {res.get('latency', 0):.2f}s").italic = True
    doc.add_paragraph("-" * 40)

doc.save(docx_path)
print(f"Updated DOCX saved to {docx_path}")
print("\nDone! Both files updated.")
