"""
RAG query pipeline for SEC 10-K filings.

Takes a question, retrieves relevant chunks from Chroma,
and sends them to Claude for a grounded answer.
"""

from pathlib import Path
from dataclasses import dataclass

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import anthropic
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from config import (
    CHROMA_COLLECTION, CHROMA_PERSIST_DIR, EMBEDDING_MODEL,
    LLM_MODEL, RETRIEVAL_K, MAX_TOKENS, SYSTEM_PROMPT_PATH,
)

# --- Module-level initialization (runs once on import) ---

client = anthropic.Anthropic()

embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

vectorstore = Chroma(
    collection_name=CHROMA_COLLECTION,
    persist_directory=CHROMA_PERSIST_DIR,
    embedding_function=embeddings,
)

system_prompt = (Path(__file__).resolve().parent / SYSTEM_PROMPT_PATH).read_text()


def retrieve(question: str, k: int = RETRIEVAL_K):
    """Embed the question and return the top-k most similar chunks from Chroma."""
    results = vectorstore.similarity_search_with_score(question, k=k)
    return results


def format_context(results) -> str:
    """Format retrieved chunks into numbered source blocks for the prompt."""
    blocks = []
    for i, (doc, score) in enumerate(results, 1):
        meta = doc.metadata
        header = (
            f"[Source {i}] "
            f"Company: {meta['company']} | "
            f"Section: {meta['section']} | "
            f"Filed: {meta['filing_date']}"
        )
        blocks.append(f"{header}\n{doc.page_content}")
    return "\n\n".join(blocks)


def query(question: str) -> dict:
    """Run the full RAG pipeline: retrieve, build prompt, call Claude."""
    results = retrieve(question)
    context = format_context(results)

    user_message = f"Here are the relevant SEC filing excerpts:\n\n{context}\n\n---\nQuestion: {question}"

    response = client.messages.create(
        model=LLM_MODEL,
        max_tokens=MAX_TOKENS,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
    )

    sources = [
        {
            "company": doc.metadata["company"],
            "section": doc.metadata["section"],
            "filing_date": doc.metadata["filing_date"],
            "chunk_index": doc.metadata["chunk_index"],
            "score": round(score, 4),
            "text_preview": doc.page_content[:100],
        }
        for doc, score in results
    ]

    return {
        "answer": response.content[0].text,
        "sources": sources,
        "model": response.model,
        "usage": {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        },
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        question = input("Question: ")

    result = query(question)

    print(f"\n{'=' * 60}")
    print(result["answer"])
    print(f"\n{'=' * 60}")
    print(f"Model: {result['model']}")
    print(f"Tokens: {result['usage']['input_tokens']} in / {result['usage']['output_tokens']} out")
    print(f"\nSources ({len(result['sources'])}):")
    for s in result["sources"]:
        print(f"  [{s['company']}] {s['section']} (filed {s['filing_date']}, chunk {s['chunk_index']}, score {s['score']})")
        print(f"    {s['text_preview']}...")
