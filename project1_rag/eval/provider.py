"""
Promptfoo provider - hooks into our RAG query() pipeline.

Promptfoo calls call_api() once per (question x config) combination.
The config (store + collection_type) comes from promptfooconfig.yaml.
"""

import os
import sys
import time
from pathlib import Path

# Promptfoo runs from eval/, but query.py uses relative paths like ./chroma_db.
# Change to the project root so everything resolves correctly.
_project_root = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, _project_root)
os.chdir(_project_root)

from query import query


def call_api(prompt, options, context):
    """Promptfoo provider entry point.

    prompt:  The rendered prompt string (just the question, since our
             prompt template is a passthrough "{{question}}").
    options: Dict with "config" key holding store/collection_type
             from promptfooconfig.yaml.
    context: Promptfoo context (vars, provider info, etc.).

    Returns a dict with "output" (answer text) and optional "metadata".
    """
    config = options.get("config", {})
    store = config.get("store", "chroma")
    collection_type = config.get("collection_type", "fixed")
    use_rerank = config.get("rerank", False)

    # Retry on rate limit (429) with backoff.
    for attempt in range(3):
        try:
            result = query(
                prompt,
                collection_type=collection_type,
                store=store,
                use_rerank=use_rerank,
            )
            break
        except Exception as e:
            if "rate_limit" in str(e) and attempt < 2:
                time.sleep(30 * (attempt + 1))
                continue
            raise

    # Return the answer as output, and attach sources + token usage as
    # metadata so they're visible in the promptfoo dashboard.
    return {
        "output": result["answer"],
        "metadata": {
            "store": store,
            "collection_type": collection_type,
            "reranked": use_rerank,
            "model": result["model"],
            "input_tokens": result["usage"]["input_tokens"],
            "output_tokens": result["usage"]["output_tokens"],
            "sources": [
                f"[{s['company']}] {s['section']} chunk {s['chunk_index']} (score {s['score']})"
                for s in result["sources"]
            ],
        },
    }
