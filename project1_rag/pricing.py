"""
Pricing data and cost computation for LLM and embedding models.

All prices in USD per million tokens (MTok).
Last verified: 2026-04-16
Sources:
  - Anthropic: https://platform.claude.com/docs/en/docs/about-claude/pricing
  - OpenAI:    https://platform.openai.com/docs/models/text-embedding-3-small
"""

PRICES = {
    "claude-sonnet-4-20250514": {
        "input_per_mtok":  3.00,
        "output_per_mtok": 15.00,
    },
    "text-embedding-3-small": {
        "input_per_mtok": 0.02,
        # No output pricing — embedding models don't generate tokens.
    },
}

_MTOK = 1_000_000


def compute_cost(
    *,
    embedding_tokens: int,
    embedding_model: str,
    llm_input_tokens: int,
    llm_output_tokens: int,
    llm_model: str,
) -> dict:
    """Compute per-query cost struct from token counts and model IDs.

    Returns a dict with token counts, per-component USD, and a total.
    Raises KeyError if a model is missing from PRICES (fail loud; prices must be
    explicit, never silently zeroed).
    """
    if embedding_model not in PRICES:
        raise KeyError(f"No pricing data for embedding model {embedding_model!r}")
    if llm_model not in PRICES:
        raise KeyError(f"No pricing data for LLM model {llm_model!r}")

    embedding_usd  = embedding_tokens  * PRICES[embedding_model]["input_per_mtok"]  / _MTOK
    llm_input_usd  = llm_input_tokens  * PRICES[llm_model]["input_per_mtok"]       / _MTOK
    llm_output_usd = llm_output_tokens * PRICES[llm_model]["output_per_mtok"]      / _MTOK
    total_usd = embedding_usd + llm_input_usd + llm_output_usd

    return {
        "embedding_tokens":  embedding_tokens,
        "llm_input_tokens":  llm_input_tokens,
        "llm_output_tokens": llm_output_tokens,
        "embedding_usd":  round(embedding_usd,  8),
        "llm_input_usd":  round(llm_input_usd,  8),
        "llm_output_usd": round(llm_output_usd, 8),
        "total_usd":      round(total_usd,      8),
    }
