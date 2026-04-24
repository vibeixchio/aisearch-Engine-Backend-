"""
Document ingestion pipeline.
Handles text chunking, metadata enrichment, embedding, and upsert into Qdrant.
"""
import logging
import asyncio
from typing import List, Dict, Any

from app.core.config import settings
from app.services.embedder import embed_texts
from app.services.vector_store import upsert_chunks

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
def chunk_text(
    text: str,
    chunk_size: int = None,
    chunk_overlap: int = None,
) -> List[str]:
    """
    Sliding-window character-level chunker.
    Splits on whitespace boundaries to avoid cutting mid-word.
    """
    chunk_size = chunk_size or settings.RAG_CHUNK_SIZE
    chunk_overlap = chunk_overlap or settings.RAG_CHUNK_OVERLAP

    words = text.split()
    chunks: List[str] = []
    current: List[str] = []
    current_len = 0

    for word in words:
        current.append(word)
        current_len += len(word) + 1  # +1 for space

        if current_len >= chunk_size:
            chunks.append(" ".join(current))
            # Roll back by overlap amount
            overlap_words: List[str] = []
            overlap_len = 0
            for w in reversed(current):
                overlap_len += len(w) + 1
                if overlap_len >= chunk_overlap:
                    break
                overlap_words.insert(0, w)
            current = overlap_words
            current_len = sum(len(w) + 1 for w in current)

    if current:
        chunks.append(" ".join(current))

    return [c for c in chunks if c.strip()]


# ---------------------------------------------------------------------------
# Metadata enrichment
# ---------------------------------------------------------------------------
def enrich_chunks(
    chunks: List[str],
    filename: str,
    workspace_id: str,
    doc_id: str,
) -> List[Dict[str, Any]]:
    """Attach metadata payload to each raw text chunk."""
    return [
        {
            "text": chunk,
            "source": filename,
            "workspace_id": workspace_id,
            "doc_id": doc_id,
            "chunk_index": i,
        }
        for i, chunk in enumerate(chunks)
    ]


# ---------------------------------------------------------------------------
# Full ingestion pipeline
# ---------------------------------------------------------------------------
async def ingest_document(
    text: str,
    filename: str,
    workspace_id: str,
    doc_id: str,
) -> Dict[str, Any]:
    """
    Full ingestion pipeline:
      1. Chunk text
      2. Enrich with metadata
      3. Embed chunks (runs in executor to avoid event loop blocking)
      4. Upsert to Qdrant

    Returns summary dict with chunk counts.
    """
    logger.info(
        "Ingesting document: filename=%s workspace=%s doc_id=%s text_len=%d",
        filename, workspace_id, doc_id, len(text),
    )

    # 1. Chunk
    raw_chunks = chunk_text(text)
    if not raw_chunks:
        logger.warning("No chunks produced for doc_id=%s", doc_id)
        return {"workspace_id": workspace_id, "doc_id": doc_id, "chunks": 0}

    # 2. Enrich
    enriched = enrich_chunks(raw_chunks, filename, workspace_id, doc_id)

    # 3. Embed — run in thread pool to avoid blocking the async loop
    texts_to_embed = [c["text"] for c in enriched]
    vectors = await asyncio.to_thread(embed_texts, texts_to_embed)

    # 4. Prepare points
    points = [
        {"vector": vectors[i], "payload": enriched[i]}
        for i in range(len(enriched))
    ]

    # 5. Upsert
    upserted = await upsert_chunks(points)

    logger.info(
        "Ingestion complete: doc_id=%s chunks=%d upserted=%d",
        doc_id, len(raw_chunks), upserted,
    )
    return {
        "workspace_id": workspace_id,
        "doc_id": doc_id,
        "filename": filename,
        "chunks": upserted,
    }
