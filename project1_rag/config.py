COMPANIES = ["AAPL", "NVDA", "DIS", "JPM", "BX", "CRWD"]
SECTIONS = ["business", "risk_factors", "management_discussion"]  # TenK attribute names

# Fixed chunking (RecursiveCharacterTextSplitter)
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 150

# Variable structure-aware chunking
VARIABLE_FLOOR = 800        # soft target minimum chunk size
VARIABLE_CEILING = 2000     # hard cap on chunk size
HEADER_MAX_LEN = 100        # blocks <= this length are candidate headers

CHROMA_COLLECTION_FIXED = "sec_filings_fixed_chunk"
CHROMA_COLLECTION_VARIABLE = "sec_filings_variable_section_chunk"
CHROMA_PERSIST_DIR = "./chroma_db"

PGVECTOR_COLLECTION_FIXED = "sec_filings_fixed_chunk"
PGVECTOR_COLLECTION_VARIABLE = "sec_filings_variable_section_chunk"

EMBEDDING_MODEL = "text-embedding-3-small"

LLM_MODEL = "claude-sonnet-4-20250514"
RETRIEVAL_K = 8
MAX_TOKENS = 1024
SYSTEM_PROMPT_PATH = "prompts/rag_v1.txt"

EVAL_QUESTIONS = [
    "What does CrowdStrike say about the risks from the July 2024 outage incident?",
    "How does Blackstone describe its business model and revenue sources?",
    "Compare the competitive risks disclosed by NVIDIA and Apple.",
    "What does Disney's MD&A say about the performance of its streaming business?",
    "Should I buy JPMorgan stock?",
]

TEXT_REPLACEMENTS = {
    "\xa0": " ",      # non-breaking space
    "\u2009": " ",    # thin space
    "\u200b": "",     # zero-width space
    "\u2003": " ",    # em space
    "\u2002": " ",    # en space
    "\r\n": "\n",     # windows line endings
}