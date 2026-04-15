"""
Ingest 10-K filings into chunks for RAG.

For each company: pull the latest 10-K, extract target sections, and split
into chunks using two strategies:
  - fixed: RecursiveCharacterTextSplitter with a fixed size/overlap
  - variable: structure-aware splitting on detected headers, with a
    paragraph merger bounded by a floor and ceiling

Each chunk set is embedded once and written to both Chroma (local) and
pgvector (Supabase), producing 4 total collections.
"""

import os
import re
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from edgar import set_identity, Company
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_postgres import PGVector
from config import (
    COMPANIES, SECTIONS, CHUNK_SIZE, CHUNK_OVERLAP,
    VARIABLE_FLOOR, VARIABLE_CEILING, HEADER_MAX_LEN,
    CHROMA_COLLECTION_FIXED, CHROMA_COLLECTION_VARIABLE,
    CHROMA_PERSIST_DIR, EMBEDDING_MODEL,
    PGVECTOR_COLLECTION_FIXED, PGVECTOR_COLLECTION_VARIABLE,
    TEXT_REPLACEMENTS,
)

set_identity("Matt Gentile mattgentile@example.com")

fixed_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)

TERMINAL_PUNCT = set('.?!:;")')
BULLET_LINE_RE = re.compile(r"^\s*(?:[\u2022\-\u2013*]|\d+\.)")
PAGE_ARTIFACT_RE = re.compile(
    r"""
    ^\s*\d{1,4}\s*$              # bare page number
    | ^\s*Part\s+[IVX]+\s*$       # "Part I", "Part II", ...
    | ^.*\|\s*.*Form\s+10-K\s*\|\s*\d+\s*$   # "Apple Inc. | 2025 Form 10-K | 5"
    """,
    re.VERBOSE | re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Section text extraction (shared by both chunkers)
# ---------------------------------------------------------------------------

def extract_section_texts(ticker: str) -> list[tuple[str, str, dict]]:
    """Pull latest 10-K and return cleaned section texts + base metadata.

    Returns a list of (section_name, cleaned_text, base_metadata) tuples.
    Chunking is done by the caller so one filing pull feeds both strategies.
    """
    company = Company(ticker)
    filing = company.get_filings(form="10-K", amendments=False).latest()
    tenk = filing.obj()

    out = []
    for section_name in SECTIONS:
        section_text = getattr(tenk, section_name, None)
        if not section_text:
            print(f"  [{ticker}] {section_name}: not available, skipping")
            continue

        section_text = str(section_text)
        for old, new in TEXT_REPLACEMENTS.items():
            section_text = section_text.replace(old, new)

        base_meta = {
            "company": company.name,
            "section": section_name,
            "filing_date": str(filing.filing_date),
        }
        out.append((section_name, section_text, base_meta))
    return out


# ---------------------------------------------------------------------------
# Fixed chunker (existing behavior)
# ---------------------------------------------------------------------------

def chunk_fixed(
    section_name: str,
    section_text: str,
    base_meta: dict,
    company_name: str,
) -> list[Document]:
    """Chunk a section using RecursiveCharacterTextSplitter (fixed size)."""
    prefixed = f"Company: {company_name} | Section: {section_name}\n{section_text}"
    docs = fixed_splitter.create_documents(
        texts=[prefixed],
        metadatas=[{**base_meta, "chunk_index": 0}],
    )
    for i, doc in enumerate(docs):
        doc.metadata["chunk_index"] = i
    return docs


# ---------------------------------------------------------------------------
# Variable chunker (structure-aware)
# ---------------------------------------------------------------------------

def _strip_page_artifacts(text: str) -> str:
    """Remove page-number/pagination lines that leak into section text."""
    kept = []
    for line in text.split("\n"):
        if PAGE_ARTIFACT_RE.match(line):
            continue
        kept.append(line)
    return "\n".join(kept)


def _split_into_blocks(text: str) -> list[str]:
    """Split on one or more blank lines; return non-empty stripped blocks."""
    return [b.strip() for b in re.split(r"\n\s*\n+", text) if b.strip()]


def _ends_terminal(block: str) -> bool:
    return bool(block) and block[-1] in TERMINAL_PUNCT


def _glue_sentence_fragments(blocks: list[str]) -> list[str]:
    """Merge blocks broken mid-sentence by the HTML-to-text conversion.

    If block N doesn't end in terminal punctuation and block N+1 starts
    with a lowercase letter, merge them with a single space. Repeats until
    stable so cascading fragments collapse correctly.
    """
    changed = True
    while changed:
        changed = False
        merged: list[str] = []
        i = 0
        while i < len(blocks):
            cur = blocks[i]
            if i + 1 < len(blocks):
                nxt = blocks[i + 1]
                if (
                    not _ends_terminal(cur)
                    and nxt
                    and nxt[0].islower()
                ):
                    merged.append(f"{cur} {nxt}")
                    i += 2
                    changed = True
                    continue
            merged.append(cur)
            i += 1
        blocks = merged
    return blocks


def _is_bullet_block(block: str) -> bool:
    lines = [l for l in block.split("\n") if l.strip()]
    if not lines:
        return False
    return all(BULLET_LINE_RE.match(l) for l in lines)


def _is_header(block: str, prev_block: str | None) -> bool:
    if "\n" in block:
        return False
    if len(block) > HEADER_MAX_LEN:
        return False
    if not block or not block[0].isupper():
        return False
    if block.replace(".", "").replace(" ", "").isdigit():
        return False
    # Second defense against sentence fragments: require prior terminal punct
    # (or that this is the first block of the section).
    if prev_block is not None and not _ends_terminal(prev_block):
        return False
    return True


def _walk_subsections(
    blocks: list[str],
) -> list[tuple[str | None, list[str]]]:
    """Group blocks into (header, [content_blocks]) subsections."""
    subsections: list[tuple[str | None, list[str]]] = []
    current_header: str | None = None
    current_content: list[str] = []
    prev_block: str | None = None

    for block in blocks:
        if _is_header(block, prev_block):
            # Flush current subsection before starting a new one.
            if current_header is not None or current_content:
                subsections.append((current_header, current_content))
            current_header = block
            current_content = []
        else:
            current_content.append(block)
        prev_block = block

    if current_header is not None or current_content:
        subsections.append((current_header, current_content))
    return subsections


def _coalesce_bullet_runs(blocks: list[str]) -> list[str]:
    """Merge bullet blocks within a subsection's content list.

    - Consecutive bullet blocks merge together.
    - A bullet block merges into the preceding non-bullet block
      (header is not in this list, so this is safe).
    - If a bullet run has no preceding non-bullet block, it stands alone.
    """
    if not blocks:
        return blocks

    result: list[str] = []
    for block in blocks:
        if _is_bullet_block(block):
            if result and not _is_bullet_block(result[-1]):
                # Merge into preceding paragraph.
                result[-1] = f"{result[-1]}\n{block}"
            elif result and _is_bullet_block(result[-1]):
                # Merge with prior bullet block.
                result[-1] = f"{result[-1]}\n{block}"
            else:
                result.append(block)
        else:
            result.append(block)
    return result


def _build_chunks_for_subsection(
    paragraphs: list[str],
    ceiling: int,
    floor: int,
) -> list[str]:
    """Greedily merge paragraphs into chunks bounded by floor/ceiling.

    - Never exceed ceiling.
    - Once the buffer reaches floor, flush at the next paragraph boundary
      instead of packing to ceiling. Keeps chunk sizes tighter.
    - Flush at end-of-subsection even if under floor; preserving header
      boundaries is the priority.
    - A single paragraph larger than ceiling is broken up with the
      fallback splitter.
    """
    chunks: list[str] = []
    current = ""
    giant_splitter = RecursiveCharacterTextSplitter(
        chunk_size=ceiling,
        chunk_overlap=0,
    )

    def flush():
        nonlocal current
        if current:
            chunks.append(current)
            current = ""

    for p in paragraphs:
        if len(p) > ceiling:
            # Giant paragraph: flush, then break the paragraph down.
            flush()
            for piece in giant_splitter.split_text(p):
                chunks.append(piece)
            continue

        if not current:
            current = p
            continue

        if len(current) + 2 + len(p) > ceiling:
            # Adding this paragraph would overflow - flush and start fresh.
            flush()
            current = p
        elif len(current) >= floor:
            # Already past the floor - flush at this natural boundary.
            flush()
            current = p
        else:
            current = f"{current}\n\n{p}"

    flush()
    return chunks


def chunk_variable(
    section_name: str,
    section_text: str,
    base_meta: dict,
    company_name: str,
) -> list[Document]:
    """Structure-aware chunker. See module docstring and the plan."""
    text = _strip_page_artifacts(section_text)
    blocks = _split_into_blocks(text)
    blocks = _glue_sentence_fragments(blocks)
    subsections = _walk_subsections(blocks)

    docs: list[Document] = []
    chunk_index = 0
    for header, content_blocks in subsections:
        content_blocks = _coalesce_bullet_runs(content_blocks)
        if not content_blocks:
            continue

        subsection_label = header if header else "-"
        prefix = (
            f"Company: {company_name} | "
            f"Section: {section_name} | "
            f"Subsection: {subsection_label}\n"
        )
        # Adjust ceiling so prefixed chunks still fit under VARIABLE_CEILING.
        effective_ceiling = max(VARIABLE_CEILING - len(prefix), VARIABLE_FLOOR)
        raw_chunks = _build_chunks_for_subsection(
            content_blocks, effective_ceiling, VARIABLE_FLOOR
        )

        for raw in raw_chunks:
            page_content = prefix + raw
            meta = {
                **base_meta,
                "subsection": subsection_label,
                "chunk_index": chunk_index,
            }
            docs.append(Document(page_content=page_content, metadata=meta))
            chunk_index += 1

    return docs


# ---------------------------------------------------------------------------
# Dual-store writer (Chroma + pgvector)
# ---------------------------------------------------------------------------

def _write_to_stores(
    docs: list[Document],
    pre_embeddings: list[list[float]],
    embeddings_model,
    chroma_collection_name: str,
    pgvector_collection_name: str,
    persist_dir: str,
    connection_string: str,
):
    """Write pre-embedded documents to both Chroma and pgvector."""
    texts = [d.page_content for d in docs]
    metadatas = [d.metadata for d in docs]
    ids = [str(i) for i in range(len(docs))]

    # --- Chroma (delete + re-add for idempotent re-runs) ---
    chroma = Chroma(
        collection_name=chroma_collection_name,
        persist_directory=persist_dir,
        embedding_function=embeddings_model,
    )
    chroma.delete_collection()
    chroma = Chroma(
        collection_name=chroma_collection_name,
        persist_directory=persist_dir,
        embedding_function=embeddings_model,
    )
    chroma._collection.add(
        ids=ids,
        embeddings=pre_embeddings,
        documents=texts,
        metadatas=metadatas,
    )
    print(f"    Chroma '{chroma_collection_name}': {len(docs)} chunks")

    # --- pgvector ---
    pgvector_store = PGVector(
        embeddings=embeddings_model,
        collection_name=pgvector_collection_name,
        connection=connection_string,
    )
    pgvector_store.add_embeddings(
        texts=texts,
        embeddings=pre_embeddings,
        metadatas=metadatas,
    )
    print(f"    pgvector '{pgvector_collection_name}': {len(docs)} chunks")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    fixed_chunks: list[Document] = []
    variable_chunks: list[Document] = []

    for ticker in COMPANIES:
        print(f"\nProcessing {ticker}...")
        sections = extract_section_texts(ticker)
        for section_name, text, base_meta in sections:
            f = chunk_fixed(section_name, text, base_meta, base_meta["company"])
            v = chunk_variable(section_name, text, base_meta, base_meta["company"])
            fixed_chunks.extend(f)
            variable_chunks.extend(v)
            print(f"  [{ticker}] {section_name}: fixed={len(f)} variable={len(v)}")

    print(f"\n{'=' * 50}")
    print(f"Fixed total:    {len(fixed_chunks)}")
    print(f"Variable total: {len(variable_chunks)}")

    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    connection_string = os.environ["SUPABASE_DB_URL"]

    # Clear pgvector tables once up front (drop_tables is global, not per-collection,
    # so we do it once before writing either collection).
    print("\nClearing pgvector tables...")
    PGVector(
        embeddings=embeddings,
        collection_name="throwaway",
        connection=connection_string,
    ).drop_tables()

    # Embed once per chunk set (2 API calls, not 4).
    print(f"Embedding {len(fixed_chunks)} fixed chunks...")
    fixed_vectors = embeddings.embed_documents(
        [d.page_content for d in fixed_chunks]
    )

    print(f"Embedding {len(variable_chunks)} variable chunks...")
    variable_vectors = embeddings.embed_documents(
        [d.page_content for d in variable_chunks]
    )

    # Write fixed chunks to both stores.
    print(f"\nWriting fixed chunks...")
    _write_to_stores(
        fixed_chunks, fixed_vectors, embeddings,
        CHROMA_COLLECTION_FIXED, PGVECTOR_COLLECTION_FIXED,
        CHROMA_PERSIST_DIR, connection_string,
    )

    # Write variable chunks to both stores.
    print(f"Writing variable chunks...")
    _write_to_stores(
        variable_chunks, variable_vectors, embeddings,
        CHROMA_COLLECTION_VARIABLE, PGVECTOR_COLLECTION_VARIABLE,
        CHROMA_PERSIST_DIR, connection_string,
    )

    print(f"\nDone. 4 collections written (2 Chroma + 2 pgvector).")
