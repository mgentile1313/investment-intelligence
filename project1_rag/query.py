"""
RAG query pipeline for SEC 10-K filings.

Takes a question, retrieves relevant chunks from Chroma or pgvector,
and sends them to Claude for a grounded answer.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import anthropic
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_postgres import PGVector
from config import (
    CHROMA_COLLECTION_FIXED, CHROMA_COLLECTION_VARIABLE,
    CHROMA_PERSIST_DIR, EMBEDDING_MODEL,
    PGVECTOR_COLLECTION_FIXED, PGVECTOR_COLLECTION_VARIABLE,
    LLM_MODEL, RETRIEVAL_K, MAX_TOKENS, SYSTEM_PROMPT_PATH,
)

# --- Module-level initialization (runs once on import) ---

client = anthropic.Anthropic()

embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

VECTORSTORES = {
    "chroma": {
        "fixed": Chroma(
            collection_name=CHROMA_COLLECTION_FIXED,
            persist_directory=CHROMA_PERSIST_DIR,
            embedding_function=embeddings,
        ),
        "variable": Chroma(
            collection_name=CHROMA_COLLECTION_VARIABLE,
            persist_directory=CHROMA_PERSIST_DIR,
            embedding_function=embeddings,
        ),
    },
    "pgvector": {
        "fixed": PGVector(
            embeddings=embeddings,
            collection_name=PGVECTOR_COLLECTION_FIXED,
            connection=os.environ.get("SUPABASE_DB_URL", ""),
        ),
        "variable": PGVector(
            embeddings=embeddings,
            collection_name=PGVECTOR_COLLECTION_VARIABLE,
            connection=os.environ.get("SUPABASE_DB_URL", ""),
        ),
    },
}

system_prompt = (Path(__file__).resolve().parent / SYSTEM_PROMPT_PATH).read_text()


def retrieve(
    question: str,
    k: int = RETRIEVAL_K,
    collection_type: str = "fixed",
    store: str = "chroma",
):
    """Embed the question and return the top-k most similar chunks."""
    vs = VECTORSTORES[store][collection_type]
    results = vs.similarity_search_with_score(question, k=k)
    return results


def format_context(results) -> str:
    """Format retrieved chunks into numbered source blocks for the prompt."""
    blocks = []
    for i, (doc, score) in enumerate(results, 1):
        meta = doc.metadata
        subsection = meta.get("subsection")
        header = (
            f"[Source {i}] "
            f"Company: {meta['company']} | "
            f"Section: {meta['section']} | "
            f"Filed: {meta['filing_date']}"
        )
        if subsection and subsection != "-":
            header += f" | Subsection: {subsection}"
        blocks.append(f"{header}\n{doc.page_content}")
    return "\n\n".join(blocks)


def query(
    question: str,
    collection_type: str = "fixed",
    store: str = "chroma",
) -> dict:
    """Run the full RAG pipeline: retrieve, build prompt, call Claude."""
    results = retrieve(question, collection_type=collection_type, store=store)
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
            "subsection": doc.metadata.get("subsection"),
            "score": round(score, 4),
            "text_preview": doc.page_content[:100],
        }
        for doc, score in results
    ]

    return {
        "answer": response.content[0].text,
        "sources": sources,
        "model": response.model,
        "store": store,
        "collection_type": collection_type,
        "usage": {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        },
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RAG query over SEC 10-K filings.")
    parser.add_argument("question", nargs="*", help="The question to ask")
    parser.add_argument(
        "--collection_type", "-c",
        choices=["fixed", "variable"],
        default="fixed",
        help="Which chunking strategy's collection to query (default: fixed)",
    )
    parser.add_argument(
        "--store", "-s",
        choices=["chroma", "pgvector"],
        default="chroma",
        help="Which vector store to query (default: chroma)",
    )
    args = parser.parse_args()

    if args.question:
        question = " ".join(args.question)
    else:
        question = input("Question: ")

    result = query(question, collection_type=args.collection_type, store=args.store)

    print(f"\n{'=' * 60}")
    print(result["answer"])
    print(f"\n{'=' * 60}")
    print(f"Store: {result['store']} | Collection: {result['collection_type']}")
    print(f"Model: {result['model']}")
    print(f"Tokens: {result['usage']['input_tokens']} in / {result['usage']['output_tokens']} out")
    print(f"\nSources ({len(result['sources'])}):")
    for s in result["sources"]:
        sub = f", subsection: {s['subsection']}" if s.get("subsection") else ""
        print(f"  [{s['company']}] {s['section']} (filed {s['filing_date']}, chunk {s['chunk_index']}, score {s['score']}{sub})")
        print(f"    {s['text_preview']}...")
