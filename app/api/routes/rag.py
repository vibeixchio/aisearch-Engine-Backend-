"""
RAG API routes — PostgreSQL backed.
POST /upload  → ingest document into Qdrant
POST /query   → hybrid RAG query with streaming SSE response
GET  /workspace/{workspace_id}/documents → list docs
DELETE /workspace/{workspace_id} → delete all docs for a workspace
"""
import json
import logging
import uuid

from fastapi import APIRouter, File, UploadFile, HTTPException, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.services.document_processor import DocumentProcessor
from app.services.ingestion import ingest_document
from app.services.rag import rag_query
from app.services.vector_store import delete_by_workspace
from app.services.cache_layer import (
    cache_get_or_fetch, cache_delete, workspace_docs_key,
)
from app.db.database import get_db
from app.db.sql_models import Document, Workspace
from app.core.config import settings
from app.core.security import get_current_user

logger = logging.getLogger(__name__)
router = APIRouter()


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class RAGQueryRequest(BaseModel):
    query: str
    workspace_id: str
    top_k: Optional[int] = None


# ---------------------------------------------------------------------------
# Upload + Ingest
# ---------------------------------------------------------------------------
@router.post("/upload")
async def upload_and_ingest(
    workspace_id: Optional[str] = None,
    file: UploadFile = File(...),
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Upload a document and ingest it into the RAG pipeline.

    Steps:
    1. Parse file (PDF / CSV / TXT)
    2. Chunk + embed + upsert to Qdrant
    3. Store record in PostgreSQL

    Returns workspace_id, doc_id, and chunk count.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    # Generate IDs
    if not workspace_id:
        workspace_id = f"ws_{uuid.uuid4().hex[:8]}"
    doc_id = uuid.uuid4().hex

    try:
        # Parse file text
        doc_text = await DocumentProcessor.process_file(file)
        if not doc_text.strip():
            raise HTTPException(status_code=422, detail="Could not extract text from file")

        # Ingest into Qdrant
        result = await ingest_document(
            text=doc_text,
            filename=file.filename,
            workspace_id=workspace_id,
            doc_id=doc_id,
        )

        # Persist record to PostgreSQL
        doc = Document(
            id=doc_id,
            workspace_id=workspace_id,
            filename=file.filename,
            content_length=len(doc_text),
            chunks_indexed=result["chunks"],
        )
        db.add(doc)

        # Invalidate document cache
        await cache_delete(workspace_docs_key(workspace_id))

        logger.info("RAG upload complete: workspace=%s doc=%s chunks=%d", workspace_id, doc_id, result["chunks"])
        return {
            "status": "ingested",
            "workspace_id": workspace_id,
            "document_id": doc_id,
            "filename": file.filename,
            "chunks_indexed": result["chunks"],
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("RAG upload failed")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


# ---------------------------------------------------------------------------
# RAG Query — SSE Stream
# ---------------------------------------------------------------------------
async def _sse_rag(query: str, workspace_id: str, top_k: Optional[int] = None):
    try:
        generator, sources = await rag_query(query, workspace_id=workspace_id, top_k=top_k)

        # First emit sources
        yield f'data: {json.dumps({"type": "sources", "data": sources})}\n\n'

        # Stream LLM chunks
        async for chunk in generator:
            yield f'data: {json.dumps({"type": "chunk", "text": chunk})}\n\n'

        yield f'data: {json.dumps({"type": "done"})}\n\n'

    except Exception as e:
        logger.exception("RAG SSE error: query=%s", query[:80])
        yield f'data: {json.dumps({"type": "error", "message": str(e)})}\n\n'


@router.post("/query")
async def rag_query_endpoint(
    req: RAGQueryRequest,
    user: dict = Depends(get_current_user),
):
    """
    Query documents in a workspace using the full RAG pipeline.

    Returns a Server-Sent Event stream:
    - `sources`: list of retrieved chunks with metadata
    - `chunk`: incremental LLM text tokens
    - `done`: stream complete
    - `error`: on failure
    """
    if not settings.GROQ_API_KEY:
        raise HTTPException(status_code=503, detail="GROQ_API_KEY not configured")

    return StreamingResponse(
        _sse_rag(req.query, req.workspace_id, req.top_k),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# Workspace document management
# ---------------------------------------------------------------------------
@router.get("/workspace/{workspace_id}/documents")
async def list_workspace_documents(
    workspace_id: str,
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """List all documents ingested into a workspace."""
    async def _fetch():
        result = await db.execute(
            select(Document)
            .where(Document.workspace_id == workspace_id)
            .order_by(Document.created_at.desc())
        )
        docs = result.scalars().all()
        return [
            {
                "id": d.id,
                "workspace_id": d.workspace_id,
                "filename": d.filename,
                "content_length": d.content_length,
                "chunks_indexed": d.chunks_indexed,
                "created_at": d.created_at.isoformat() if d.created_at else None,
            }
            for d in docs
        ]

    return await cache_get_or_fetch(
        workspace_docs_key(workspace_id),
        _fetch,
        ttl=settings.REDIS_WORKSPACE_TTL,
    )


@router.delete("/workspace/{workspace_id}")
async def delete_workspace_rag(
    workspace_id: str,
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Delete all vectors and document records for a workspace."""
    await delete_by_workspace(workspace_id)

    # Delete from PostgreSQL
    result = await db.execute(
        select(Document).where(Document.workspace_id == workspace_id)
    )
    docs = result.scalars().all()
    for doc in docs:
        await db.delete(doc)

    # Invalidate cache
    await cache_delete(workspace_docs_key(workspace_id))

    return {"status": "deleted", "workspace_id": workspace_id}
