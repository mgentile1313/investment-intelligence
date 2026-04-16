"""
Convert the golden dataset JSONL into a promptfoo-compatible YAML test file.

Run this whenever you update the golden dataset:
    python load_tests.py

Reads golden_dataset (1).jsonl and writes tests.yaml with assertions
per question category:
  - all categories:  llm-rubric comparing output to the ideal answer
  - out_of_scope:    + not-contains guardrails for recommendation language
"""

import json
import yaml
from pathlib import Path

GOLDEN_DATASET = Path(__file__).resolve().parent / "golden_dataset (1).jsonl"
OUTPUT = Path(__file__).resolve().parent / "tests.yaml"

OUT_OF_SCOPE_GUARDRAILS = [
    "I recommend",
    "you should buy",
    "you should invest",
    "I would suggest buying",
    "I advise",
]

RUBRIC = """\
You are grading a RAG system that answers questions about SEC 10-K filings.

IMPORTANT CONTEXT: The filings in this system are real and were filed in 2025-2026.
These are NOT future dates. The system has access to:
- Apple Inc. 10-K filed 2025-10-31 (fiscal year ending September 2025)
- NVIDIA Corp 10-K filed 2026-02-25 (fiscal year ending January 2026)
- Walt Disney Co 10-K filed 2025-11-13 (fiscal year ending September 2025)
- JPMorgan Chase 10-K filed 2026-02-13 (fiscal year ending December 2025)
- Blackstone Inc. 10-K filed 2026-02-27 (fiscal year ending December 2025)
- CrowdStrike Holdings 10-K filed 2026-03-05 (fiscal year ending January 2026)
Do NOT penalize answers for referencing data from these filings. Dates like
"fiscal 2025", "fiscal 2026", "September 2025", etc. are valid and expected.

Reference answer (written by a human expert):
{{ideal_answer}}

Grade the model's output against this reference. Consider:
1. Factual accuracy - are the specific numbers, names, and claims correct?
2. Completeness - does it cover the key points from the reference?
3. Groundedness - does it stick to information from the filings, or hallucinate?

Score as a float between 0 and 1:
- 1.0: Covers all key points accurately, no hallucination
- 0.7: Mostly accurate, minor omissions
- 0.4: Partially correct, significant gaps or minor inaccuracies
- 0.0: Wrong, hallucinated, or failed to answer appropriately"""


def generate():
    tests = []

    with open(GOLDEN_DATASET) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)

            test = {
                "description": f"[{row.get('category', '')}] {row['question'][:80]}",
                "vars": {
                    "question": row["question"],
                    "ideal_answer": row["ideal_answer"],
                    "category": row.get("category", ""),
                    "company": row.get("company", ""),
                    "difficulty": row.get("difficulty", ""),
                },
                "assert": [
                    {"type": "llm-rubric", "value": RUBRIC},
                ],
            }

            if row.get("category") == "out_of_scope":
                for phrase in OUT_OF_SCOPE_GUARDRAILS:
                    test["assert"].append({
                        "type": "not-contains",
                        "value": phrase,
                    })

            tests.append(test)

    with open(OUTPUT, "w") as f:
        yaml.dump(tests, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    print(f"Generated {len(tests)} test cases → {OUTPUT.name}")


if __name__ == "__main__":
    generate()
