"""
Compile the GroundedRBIAdvisor module using BootstrapFewShot.

Run once (or whenever trainset changes):
    py -m generation.compile_advisor

Saves compiled state to generation/compiled_advisor.json
"""

import re
import os
import sys

import dspy
from dspy.teleprompt import BootstrapFewShot

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from generation.dspy_advisor import GroundedRBIAdvisor
from generation.trainset import trainset


def metric(example, pred, trace=None):
    """
    Metric for BootstrapFewShot:
    (a) Answer contains a valid [N] citation OR an explicit non-coverage statement
    (b) Answer contains no leaked tags/fences
    """
    answer = pred.answer if hasattr(pred, 'answer') else ""

    # (a) Has citation or non-coverage statement
    has_citation = bool(re.search(r'\[\d+\]', answer))
    non_coverage_phrases = [
        "not covered", "not stated", "not detailed",
        "not in these circulars", "not in the retrieved",
        "not contain", "not available", "not specified",
        "route this query", "relevant department",
    ]
    has_non_coverage = any(phrase in answer.lower() for phrase in non_coverage_phrases)

    if not has_citation and not has_non_coverage:
        return False

    # (b) No leaked tags
    has_leaked = bool(re.search(r'<Thinking>|```|<FinalAnswer>', answer, re.IGNORECASE))
    if has_leaked:
        return False

    return True


def main():
    print("Configuring DSPy with Groq...")
    lm = dspy.LM(
        f"groq/{config.GROQ_DEFAULT_MODEL}",
        api_key=config.GROQ_API_KEY,
        api_base="https://api.groq.com/openai/v1",
        temperature=0.1,
    )
    dspy.configure(lm=lm)

    print("Instantiating GroundedRBIAdvisor...")
    module = GroundedRBIAdvisor()

    print(f"Compiling with BootstrapFewShot (trainset size={len(trainset)})...")
    optimizer = BootstrapFewShot(
        metric=metric,
        max_bootstrapped_demos=4,
        max_labeled_demos=5,
    )

    compiled = optimizer.compile(module, trainset=trainset)

    # Save compiled state
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "compiled_advisor.json")
    compiled.save(out_path)
    print(f"Compiled advisor saved to {out_path}")


if __name__ == "__main__":
    main()
