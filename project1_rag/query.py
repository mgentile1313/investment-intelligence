"""
RAG query pipeline for SEC 10-K filings.

Takes a question, retrieves relevant chunks from Chroma or pgvector,
optionally reranks them with a cross-encoder, and sends them to Claude
for a grounded answer.
"""

import os
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import anthropic
import tiktoken
from sentence_transformers import CrossEncoder
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_postgres import PGVector
from config import (
    CHROMA_COLLECTION_FIXED, CHROMA_COLLECTION_VARIABLE,
    CHROMA_PERSIST_DIR, EMBEDDING_MODEL,
    PGVECTOR_COLLECTION_FIXED, PGVECTOR_COLLECTION_VARIABLE,
    LLM_MODEL, RETRIEVAL_K, RERANK_INITIAL_K, RERANK_MODEL,
    MAX_TOKENS, SYSTEM_PROMPT_PATH,
)
from pricing import compute_cost

# Module-level init (runs once on import)

client = anthropic.Anthropic()

embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

_embed_encoder = tiktoken.encoding_for_model(EMBEDDING_MODEL)

reranker = CrossEncoder(RERANK_MODEL)

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


def rerank(question: str, results, k: int = RETRIEVAL_K):
    """Score each (question, chunk) pair with the cross-encoder and keep top k."""
    pairs = [[question, doc.page_content] for doc, _score in results]
    scores = reranker.predict(pairs)
    # Pair each result with its reranker score, sort descending, take top k.
    scored = list(zip(results, scores))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [result for result, _score in scored[:k]]


def retrieve(
    question: str,
    k: int = RETRIEVAL_K,
    collection_type: str = "fixed",
    store: str = "chroma",
    use_rerank: bool = False,
):
    """Embed the question and return the top-k most similar chunks.

    If use_rerank=True, retrieves RERANK_INITIAL_K chunks first, then
    reranks with the cross-encoder and returns the top k.
    """
    initial_k = RERANK_INITIAL_K if use_rerank else k
    vs = VECTORSTORES[store][collection_type]
    results = vs.similarity_search_with_score(question, k=initial_k)

    if use_rerank:
        results = rerank(question, results, k=k)

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
    collection_type: str = "variable",
    store: str = "pgvector",
    use_rerank: bool = False,
) -> dict:
    """Run the full RAG pipeline: retrieve, (optionally rerank), build prompt, call Claude."""
    start = time.perf_counter()

    embed_tokens = len(_embed_encoder.encode(question))

    results = retrieve(
        question,
        collection_type=collection_type,
        store=store,
        use_rerank=use_rerank,
    )
    context = format_context(results)

    user_message = f"Here are the relevant SEC filing excerpts:\n\n{context}\n\n---\nQuestion: {question}"

    response = client.messages.create(
        model=LLM_MODEL,
        max_tokens=MAX_TOKENS,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
    )

    cost = compute_cost(
        embedding_tokens=embed_tokens,
        embedding_model=EMBEDDING_MODEL,
        llm_input_tokens=response.usage.input_tokens,
        llm_output_tokens=response.usage.output_tokens,
        llm_model=LLM_MODEL,
    )

    sources = [
        {
            "company": doc.metadata["company"],
            "section": doc.metadata["section"],
            "filing_date": doc.metadata["filing_date"],
            "chunk_index": doc.metadata["chunk_index"],
            "subsection": doc.metadata.get("subsection"),
            "score": round(score, 4),
            "text_preview": doc.page_content[:150],
        }
        for doc, score in results
    ]

    latency_s = round(time.perf_counter() - start, 3)

    return {
        "answer": response.content[0].text,
        "sources": sources,
        "model": response.model,
        "store": store,
        "collection_type": collection_type,
        "reranked": use_rerank,
        "cost": cost,
        "latency_s": latency_s,
    }


if __name__ == "__main__":
    import argparse

    class _HelpFormatter(
        argparse.ArgumentDefaultsHelpFormatter,
        argparse.RawDescriptionHelpFormatter,
    ):
        pass

    parser = argparse.ArgumentParser(
        description="RAG query over SEC 10-K filings.",
        formatter_class=_HelpFormatter,
        epilog=(
            "Examples:\n"
            "  query.py \"What is Apple's revenue?\"\n"
            "  query.py -v \"How many employees does Apple have?\"\n"
            "  query.py --store chroma --rerank \"Key risks for Microsoft?\"\n"
        ),
    )
    parser.add_argument("question", nargs="*", help="The question to ask")
    parser.add_argument(
        "-c", "--collection-type",
        choices=["fixed", "variable"],
        default="variable",
        help="Chunking strategy's collection to query",
    )
    parser.add_argument(
        "-s", "--store",
        choices=["chroma", "pgvector"],
        default="pgvector",
        help="Vector store to query",
    )
    parser.add_argument(
        "-r", "--rerank",
        action="store_true",
        help="Rerank results with cross-encoder (retrieves 24, keeps top 8)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show latency, cost, scores, and retrieved chunks after the answer",
    )
    args = parser.parse_args()

    if args.question:
        question = " ".join(args.question)
    else:
        question = input("Question: ")

    result = query(
        question,
        collection_type=args.collection_type,
        store=args.store,
        use_rerank=args.rerank,
    )

    c = result["cost"]
    rerank_label = " | Reranked" if result["reranked"] else ""

    print(f"\n{'=' * 60}")
    print(result["answer"])
    print(f"{'=' * 60}")

    if not args.verbose:
        print(
            f"Store: {result['store']} | "
            f"Collection: {result['collection_type']}{rerank_label} | "
            f"Cost: ${c['total_usd']:.6f} | "
            f"Latency: {result['latency_s']}s"
        )
    else:
        print(f"\nLatency: {result['latency_s']}s")
        print(
            f"\nCost: {c['llm_input_tokens']} in / {c['llm_output_tokens']} out "
            f"/ {c['embedding_tokens']} embed tokens  |  ${c['total_usd']:.6f}"
        )
        print(
            f"\nStore: {result['store']} | "
            f"Collection: {result['collection_type']}{rerank_label} | "
            f"Model: {result['model']}"
        )
        print("\nScores:")
        print("  " + "  ".join(f"[{i}] {s['score']}" for i, s in enumerate(result["sources"], 1)))
        print(f"\nChunks ({len(result['sources'])}):")
        for i, s in enumerate(result["sources"], 1):
            sub = f" | Subsection: {s['subsection']}" if s.get("subsection") else ""
            print(f"  [{i}] {s['company']} | {s['section']} | {s['filing_date']} | chunk {s['chunk_index']}{sub}")
            print(f"      {s['text_preview']}...")
