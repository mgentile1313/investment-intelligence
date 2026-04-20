# SEC 10-K RAG Pipeline

I built this to learn how RAG design choices actually affect answer quality — not by intuition, but by measuring. It's a retrieval-augmented generation pipeline over SEC 10-K filings that I set up with multiple chunking strategies, two vector stores, and optional reranking, then wired up an eval harness to compare them all against the same 30-question golden dataset.

The idea: instead of guessing whether structure-aware chunking beats naive fixed-size chunking, or whether reranking is worth the latency, I can just run the eval and see.

**Companies:** Apple, NVIDIA, Disney, JPMorgan, Blackstone, CrowdStrike

**Sections per filing:** Business, Risk Factors, MD&A

## Architecture

```
     INGEST (ingest.py)            QUERY (query.py)               EVAL (eval/)
     ──────────────────            ────────────────               ──────────────
  10-K filings (SEC EDGAR)       User question                  Golden dataset
         │                             │                        (30 questions)
         ▼                             ▼                              │
  Extract 3 sections              Embed question                     ▼
  per company (edgartools)        (OpenAI)                     For each question
         │                             │                       x 5 configs:
         ▼                             ▼                         -> call query()
  Chunk with TWO strategies    Retrieve top-K chunks             -> LLM judges
  ┌─────────┬──────────┐       from selected store                 output vs.
  │  Fixed  │ Variable │       and collection                      ideal answer
  │ (1500c) │ (800-    │             │                              │
  │         │  2000c)  │      [optional rerank]                     ▼
  └────┬────┴────┬─────┘      (cross-encoder)               Dashboard: scores,
       │         │                   │                       costs, latency
       ▼         ▼                   ▼                       per configuration
  Embed ONCE per set          Format + call Claude
  (OpenAI, 2 API calls)             │
       │         │                   ▼
       ▼         ▼              Answer + sources
  Write to BOTH stores         + cost + latency
  ┌────────┬──────────┐
  │ Chroma │ pgvector │
  │(local) │(Supabase)│
  └────────┴──────────┘
```

**The 5 configs I compare:**

| Config | Store | Chunking | Rerank |
|--------|-------|----------|--------|
| chroma-fixed | Chroma (local) | Fixed (1500 chars) | No |
| chroma-variable | Chroma (local) | Structure-aware | No |
| pgvector-fixed | pgvector (Supabase) | Fixed (1500 chars) | No |
| pgvector-variable | pgvector (Supabase) | Structure-aware | No |
| pgvector-variable-reranked | pgvector (Supabase) | Structure-aware | Yes |

## Why I made the decisions I did

### Two vector stores

Chroma is local and file-backed — no setup, great for iterating quickly. pgvector is on Supabase — real PostgreSQL, closer to what you'd use in production. I write to both during ingestion from a single embedding pass, so the vectors are byte-identical. That way if retrieval results differ between them, I know it's the index, not the input.

### Two chunking strategies

**Fixed** is the baseline: LangChain's `RecursiveCharacterTextSplitter` at 1500 chars / 150 overlap. Treats text as a blob.

**Variable** is where the real work is (~200 lines in `ingest.py`). It strips page artifacts, glues sentence fragments that HTML-to-text conversion breaks apart, detects subsection headers, merges bullet lists with their preceding paragraphs, and packs paragraphs with a soft floor of 800 chars and a hard ceiling of 2000. The chunks end up respecting the document's actual structure.

My bet was that respecting subsection boundaries would improve retrieval. The variable chunker is significantly more code — if the eval shows it doesn't help, I'd delete it. That's the whole point of running both.

### Reranking

By default, retrieval is bi-encoder: question and chunks are embedded separately, similarity is cosine distance. The optional cross-encoder reranker (`ms-marco-MiniLM-L-6-v2`) reads the question and each chunk together to produce a better relevance score. I fetch 24 candidates and keep the top 8.

It adds maybe 0.3-0.5s and costs nothing (runs locally). Whether it's worth that latency is what the eval is for.

### Evaluation with promptfoo

I wrote 30 test cases by hand — factual lookups, cross-company comparisons, analytical questions, and a few out-of-scope ones like "Should I buy NVIDIA stock?" Each has an ideal answer I wrote myself.

Promptfoo runs all 5 configs against all 30 questions, and Claude grades each output against the ideal using an `llm-rubric`. Out-of-scope questions also have hard `not-contains` guardrails on phrases like "I recommend" — soft grading for quality, hard checks for safety.

One thing I had to deal with: the grading rubric needs to explicitly tell the judge that the 2025-2026 filing dates are real. Otherwise Claude's grader flags correct answers as hallucinations because those dates are after its training cutoff.

### Cost and latency tracking

Every `query()` call returns token counts, dollar costs (embedding + LLM input + LLM output), and latency. All of it flows into the promptfoo dashboard. I wanted to compare configs on quality-per-dollar, not just raw quality.

## File Layout

```
config.py              # All configuration: companies, chunking params, model IDs,
│                      #   collection names, text replacements
ingest.py              # Fetch 10-Ks -> chunk (fixed + variable) -> embed -> write
│                      #   to Chroma + pgvector (4 collections total)
query.py               # RAG pipeline + CLI entry point
pricing.py             # Per-query cost computation (USD per MTok)
explore_10k.py         # Scratch notebook where I figured out the edgartools API
prompts/
└── rag_v1.txt         # System prompt (short — analyst role, grounding, cite sources)
chroma_db/             # Local Chroma persistence (gitignored)
eval/
├── promptfooconfig.yaml   # 5 provider configs
├── golden_dataset.jsonl   # 30 test cases with ideal answers
├── load_tests.py          # Converts golden dataset -> tests.yaml
├── tests.yaml             # Generated — don't hand-edit
├── provider.py            # Adapter between promptfoo and query()
└── run_eval.sh            # One-command eval runner with rate-limit pacing
```

## Setup

**Prerequisites:** Python 3.12+, a Supabase project with pgvector, API keys for Anthropic and OpenAI.

Create a `.env` in the repo root:

```
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
SUPABASE_DB_URL=postgresql://postgres:...@db.xxx.supabase.co:5432/postgres
```

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install anthropic openai tiktoken python-dotenv \
    langchain langchain-openai langchain-chroma langchain-postgres langchain-text-splitters \
    edgartools sentence-transformers pyyaml pgvector chromadb
```

**EDGAR identity:** SEC requires a User-Agent with your name + email per their fair access policy. Update `set_identity()` in `ingest.py` before running ingestion.

## How to Run

### 1. Ingest (one-time)

```bash
python ingest.py
```

Takes ~5-10 minutes. Creates 4 collections (2 chunking strategies x 2 stores). Safe to re-run — it drops and recreates everything.

### 2. Query

```bash
python query.py "What does CrowdStrike say about the July 2024 outage?"
python query.py -v "How many employees does Apple have?"
python query.py --store chroma --collection-type fixed "Key risks for NVIDIA?"
python query.py --rerank "Compare Disney and Apple's revenue"
```

Default is pgvector + variable chunking + no rerank. `-v` shows latency, cost breakdown, similarity scores, and the retrieved chunks.

### 3. Eval

```bash
cd eval
./run_eval.sh                      # full: 30 questions x 5 configs
./run_eval.sh --filter-first-n 3   # smoke test
promptfoo view                     # open the dashboard
```

## Models and Costs

| Component | Model | Cost |
|-----------|-------|------|
| Embedding | `text-embedding-3-small` (OpenAI) | $0.02 / M tokens |
| LLM | `claude-sonnet-4-20250514` (Anthropic) | $3 / M input, $15 / M output |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Free (local) |

A typical query costs ~$0.006. A full eval run (150 queries + 150 grading calls) comes out to roughly $2.40.

## Known Limitations

- **Ingestion drops everything and recreates.** Fine for 6 companies, wouldn't scale without per-document tracking and incremental re-embedding.
- **LLM grading isn't perfectly reproducible** — Claude's scores can drift slightly between runs. The rubric helps but doesn't eliminate it.
- **Only one embedding model.** I vary the chunking and the store but not the embeddings themselves. Adding a second embedding model would be a natural next experiment.
- **Eval pacing is slow.** Running serial with 15s delays to stay under rate limits means a full eval takes ~45 minutes. Not a code problem, just an API tier constraint.
