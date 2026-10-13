from generation.dspy_advisor import generate_dspy_answer
import json

if __name__ == "__main__":
    queries = [
        "What is the eligible limit for inclusion of Perpetual Debt Instruments (PDIs) issued in foreign currency in Additional Tier 1 (AT1) capital?",
        "Which specific circular regarding Perpetual Debt Instruments (PDI) in Additional Tier 1 Capital was repealed on September 29, 2025?",
        "Explain the tier 1 capital requirements under Basel III and how PDIs are factored in."
    ]
    
    for q in queries:
        print(f"Testing query: {q}")
        try:
            res = generate_dspy_answer(q)
            print("Answer:", res.get("answer"))
            print("Retrieved chunks:", len(res.get("retrieved_chunks", [])))
            print("Model:", res.get("model"))
            print("Latency:", res.get("latency"))
        except Exception as e:
            print(f"Error: {e}")
        print("-" * 80)
