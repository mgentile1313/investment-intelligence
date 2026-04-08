"""
Ingest 10-K filings into chunks for RAG.

For each company: pull the latest 10-K, extract target sections,
and split into chunks with metadata.
"""

from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from edgar import set_identity, Company
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from config import (
    COMPANIES, SECTIONS, CHUNK_SIZE, CHUNK_OVERLAP,
    CHROMA_COLLECTION, CHROMA_PERSIST_DIR, EMBEDDING_MODEL,
    TEXT_REPLACEMENTS,
)

set_identity("Matt Gentile mattgentile@example.com")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)


def extract_sections(ticker: str) -> list[dict]:
    """Pull latest 10-K for a ticker and return chunked sections with metadata."""
    company = Company(ticker)
    filing = company.get_filings(form="10-K", amendments=False).latest()
    tenk = filing.obj()

    chunks = []
    for section_name in SECTIONS:
        section_text = getattr(tenk, section_name, None)
        if not section_text:
            print(f"  [{ticker}] {section_name}: not available, skipping")
            continue

        section_text = str(section_text)
        for old, new in TEXT_REPLACEMENTS.items():
            section_text = section_text.replace(old, new)
        section_text = f"Company: {company.name} | Section: {section_name}\n{section_text}"
        docs = splitter.create_documents(
            texts=[section_text],
            metadatas=[{
                "company": company.name,
                "section": section_name,
                "filing_date": str(filing.filing_date),
                "chunk_index": 0,
            }],
        )
        for i, doc in enumerate(docs):
            doc.metadata["chunk_index"] = i
        chunks.extend(docs)
        print(f"  [{ticker}] {section_name}: {len(section_text):,} chars → {len(docs)} chunks")

    return chunks


if __name__ == "__main__":
    # Step 1: Extract and chunk all filings
    all_chunks = []
    for ticker in COMPANIES:
        print(f"\nProcessing {ticker}...")
        ticker_chunks = extract_sections(ticker)
        all_chunks.extend(ticker_chunks)

    print(f"\n{'=' * 50}")
    print(f"Total chunks: {len(all_chunks)}")
    print(f"\nSample chunk:")
    print(f"  metadata: {all_chunks[0].metadata}")
    print(f"  content:  {all_chunks[0].page_content}")
    print(f"  length:  {len(all_chunks[0].page_content):,} chars")

    # Step 2: Create the embedding model
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    # Step 3: Store chunks in Chroma (embeds automatically)
    print(f"\nEmbedding and storing {len(all_chunks)} chunks in Chroma...")
    vectorstore = Chroma.from_documents(
        documents=all_chunks,
        embedding=embeddings,
        collection_name=CHROMA_COLLECTION,
        persist_directory=CHROMA_PERSIST_DIR,
    )
    print(f"Done. Collection '{CHROMA_COLLECTION}' saved to {CHROMA_PERSIST_DIR}")
