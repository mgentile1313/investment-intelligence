COMPANIES = ["AAPL", "NVDA", "DIS", "JPM", "BX"]  # Pick 5 you care about
SECTIONS = ["business", "risk_factors", "management_discussion"]  # TenK attribute names
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 150
CHROMA_COLLECTION = "sec_filings_fixed_chunk"
CHROMA_PERSIST_DIR = "./chroma_db"
EMBEDDING_MODEL = "text-embedding-3-small"

LLM_MODEL = "claude-sonnet-4-20250514"
RETRIEVAL_K = 8
MAX_TOKENS = 1024
SYSTEM_PROMPT_PATH = "prompts/rag_v1.txt"

TEXT_REPLACEMENTS = {
    "\xa0": " ",      # non-breaking space
    "\u2009": " ",    # thin space
    "\u200b": "",     # zero-width space
    "\u2003": " ",    # em space
    "\u2002": " ",    # en space
    "\r\n": "\n",     # windows line endings
}