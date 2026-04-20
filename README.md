# Investment Intelligence

I'm building an AI-powered investment intelligence platform, adding capabilities over 14 weeks. It starts as a simple question-answering tool over SEC filings and grows into a multi-agent due diligence system.

The end goal: a system that pulls 10-K narratives, tracks institutional holdings (13F), monitors insider trading (Form 4), scans private placement offerings (Form D), analyzes material events (8-K) in real time, and produces structured investment memos — with observability and evaluation throughout.

Each project is a layer on top of the last, not a standalone tutorial. The repo tells one story.

## Why SEC / investment data

I picked this domain because the data is free, dense, and structured enough to do real work with. SEC filings go way beyond 10-Ks — 13F institutional holdings, Form 4 insider trades, Form D private offerings, 8-K material events, Form ADV adviser disclosures, proxy statements. Financial data also has verifiable ground truth (numbers are right or wrong), which makes evaluation meaningful rather than subjective.

## Projects

| # | What it adds | Status |
|---|-------------|--------|
| 1 | [**SEC 10-K RAG Pipeline**](project1_rag/) — RAG over 10-K filings with multiple chunking strategies, dual vector stores, cross-encoder reranking, LLM-judged eval, cost tracking | Done |

## Setup

Python 3.12+. Each project has its own README with specific setup instructions. API keys go in a `.env` at the repo root — see individual project READMEs for which vars are needed.
