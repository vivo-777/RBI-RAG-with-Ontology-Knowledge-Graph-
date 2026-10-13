import json
import os

def format_results():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(base_dir, "groq_40_results.json")
    output_file = os.path.join(base_dir, "groq_12_results.md")

    if not os.path.exists(input_file):
        print("Results file not found.")
        return

    with open(input_file, "r") as f:
        results = json.load(f)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("# Groq 12 Queries Results\n\n")
        f.write("> **Note:** The script was paused after Query 12 due to Groq free-tier payload limits (prompt exceeded 6,000 tokens).\n\n")
        
        for idx, res in enumerate(results):
            f.write(f"### Query {idx + 1}\n")
            f.write(f"**Question:** {res.get('query', '')}\n\n")
            f.write(f"**Answer:** {res.get('answer', '')}\n\n")
            f.write(f"*Latency: {res.get('latency', 0):.2f}s*\n\n")
            f.write("---\n\n")

    print(f"Successfully wrote 12 formatted results to {output_file}")

if __name__ == "__main__":
    format_results()
