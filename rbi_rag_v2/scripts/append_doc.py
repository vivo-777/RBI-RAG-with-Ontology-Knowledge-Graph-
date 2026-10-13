with open('TECHNICAL_DOCUMENTATION.md', 'a', encoding='utf-8') as f:
    f.write("\n## 5. DSPy & Groq Optimization Strategies\n\n")
    f.write("To handle Groq's extremely strict Tokens-Per-Minute (TPM) and Tokens-Per-Day (TPD) limits on the free tier, the system employs the following optimizations:\n")
    f.write("* **Throttled Execution:** An artificial 35-second pause is injected between evaluation queries to ensure the rolling 60-second window does not exceed 12,000 TPM.\n")
    f.write("* **Context Pruning:** Retaining only the top 3-8 reranked chunks drastically reduces prompt size, preventing 'Request Too Large' errors on the 70B model.\n")
    f.write("* **Model Routing for Evaluation:** To bypass the 100,000 TPD limit on llama-3.3-70b-versatile, the evaluation framework strictly routes factual verification (Faithfulness/Precision metrics) to the llama-3.1-8b-instant model. This reduces token consumption per query from ~55,000 tokens down to ~5,000 tokens on the 70B model, preserving daily quota while maintaining high precision for the actual answer generation.\n")
    f.write("* **Disabled Self-Consistency:** DSPy's default multi-vote generation (temperature=0.7) is disabled to prevent 3x token multiplication per query.\n")
