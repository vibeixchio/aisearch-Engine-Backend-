"""
Workspace routes — full CRUD, invitation system, member management, document upload,
workspace chat, and workspace graph.
All backed by PostgreSQL (Supabase) + Redis caching.
"""
import uuid
import json
import logging
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, File, UploadFile, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, EmailStr
from typing import Optional, List
from sqlalchemy import select, and_, or_, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.db.database import get_db
from app.db.sql_models import (
    Workspace, WorkspaceMember, Document, Invitation, SearchHistory, User,
)
from app.core.security import get_current_user
from app.core.config import settings
from app.services.document_processor import DocumentProcessor
from app.services.ingestion import ingest_document
from app.api.routes.rag import _sse_rag
from app.services.cache_layer import (
    cache_get_or_fetch, cache_delete, cache_delete_pattern,
    workspace_key, workspace_list_key, workspace_docs_key,
)

logger = logging.getLogger(__name__)
router = APIRouter()

# Max upload size: 50 MB
MAX_UPLOAD_BYTES = 50 * 1024 * 1024


# ---------------------------------------------------------------------------
# Request / Response Models
# ---------------------------------------------------------------------------
class CreateWorkspaceRequest(BaseModel):
    name: str = "New Workspace"


class UpdateWorkspaceRequest(BaseModel):
    name: Optional[str] = None


class ShareWorkspaceRequest(BaseModel):
    email: EmailStr
    role: str = "viewer"


class InvitationActionRequest(BaseModel):
    action: str  # "accept" or "reject"


class UpdateMemberRoleRequest(BaseModel):
    role: str  # "editor" or "viewer"


class WorkspaceChatRequest(BaseModel):
    query: str
    workspace_id: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
async def _get_workspace_or_404(
    workspace_id: str, db: AsyncSession
) -> Workspace:
    result = await db.execute(
        select(Workspace)
        .options(selectinload(Workspace.members))
        .where(Workspace.id == workspace_id)
    )
    ws = result.scalar_one_or_none()
    if not ws:
        raise HTTPException(status_code=404, detail="Workspace not found")
    return ws


def _is_member(ws: Workspace, user_id: str) -> bool:
    """Check if user is owner or member of workspace."""
    if ws.user_id == user_id:
        return True
    return any(m.user_id == user_id for m in ws.members)


def _is_owner(ws: Workspace, user_id: str) -> bool:
    return ws.user_id == user_id


def _get_role(ws: Workspace, user_id: str) -> Optional[str]:
    if ws.user_id == user_id:
        return "owner"
    for m in ws.members:
        if m.user_id == user_id:
            return m.role
    return None


def _ws_to_dict(ws: Workspace, docs_count: int = 0) -> dict:
    return {
        "id": ws.id,
        "name": ws.name,
        "user_id": ws.user_id,
        "created_at": ws.created_at.isoformat() if ws.created_at else None,
        "updated_at": ws.updated_at.isoformat() if ws.updated_at else None,
        "members": [
            {"user_id": m.user_id, "role": m.role}
            for m in ws.members
        ],
        "documents_count": docs_count,
    }


# ===========================================================================
# INVITATION ROUTES — MUST come before /{workspace_id} to avoid shadowing!
# ===========================================================================

# ---------------------------------------------------------------------------
# GET /invitations/pending — List Pending Invitations for Current User
# ---------------------------------------------------------------------------
@router.get("/invitations/pending")
async def list_pending_invitations(
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
):
    offset = (page - 1) * per_page

    # Count total
    count_result = await db.execute(
        select(func.count(Invitation.id)).where(
            and_(
                Invitation.invitee_email == user["email"],
                Invitation.status == "pending",
                or_(
                    Invitation.expires_at.is_(None),
                    Invitation.expires_at > datetime.now(timezone.utc),
                ),
            )
        )
    )
    total = count_result.scalar() or 0

    result = await db.execute(
        select(Invitation)
        .where(
            and_(
                Invitation.invitee_email == user["email"],
                Invitation.status == "pending",
                or_(
                    Invitation.expires_at.is_(None),
                    Invitation.expires_at > datetime.now(timezone.utc),
                ),
            )
        )
        .order_by(Invitation.created_at.desc())
        .offset(offset)
        .limit(per_page)
    )
    invitations = result.scalars().all()

    return {
        "invitations": [
            {
                "id": inv.id,
                "workspace_id": inv.workspace_id,
                "inviter_id": inv.inviter_id,
                "role": inv.role,
                "created_at": inv.created_at.isoformat() if inv.created_at else None,
                "expires_at": inv.expires_at.isoformat() if inv.expires_at else None,
            }
            for inv in invitations
        ],
        "pagination": {"page": page, "per_page": per_page, "total": total},
    }


# ---------------------------------------------------------------------------
# POST /invitations/{invitation_id}/respond — Accept or Reject
# ---------------------------------------------------------------------------
@router.post("/invitations/{invitation_id}/respond")
async def respond_to_invitation(
    invitation_id: str,
    req: InvitationActionRequest,
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Invitation).where(Invitation.id == invitation_id)
    )
    invitation = result.scalar_one_or_none()
    if not invitation:
        raise HTTPException(status_code=404, detail="Invitation not found")

    if invitation.invitee_email != user["email"]:
        raise HTTPException(status_code=403, detail="This invitation is not for you")

    if invitation.status != "pending":
        raise HTTPException(status_code=400, detail=f"Invitation already {invitation.status}")

    if invitation.expires_at and invitation.expires_at < datetime.now(timezone.utc):
        invitation.status = "expired"
        raise HTTPException(status_code=400, detail="Invitation has expired")

    if req.action == "accept":
        invitation.status = "accepted"
        # Add user as member
        member = WorkspaceMember(
            workspace_id=invitation.workspace_id,
            user_id=user["id"],
            role=invitation.role,
        )
        db.add(member)

        # Invalidate caches
        await cache_delete(workspace_key(invitation.workspace_id))
        await cache_delete(workspace_list_key(user["id"]))

        logger.info("Invitation accepted: %s → workspace %s", user["id"], invitation.workspace_id)
        return {"status": "accepted", "workspace_id": invitation.workspace_id}

    elif req.action == "reject":
        invitation.status = "rejected"
        logger.info("Invitation rejected: %s → workspace %s", user["id"], invitation.workspace_id)
        return {"status": "rejected"}

    else:
        raise HTTPException(status_code=400, detail="Action must be 'accept' or 'reject'")


# ===========================================================================
# WORKSPACE CRUD ROUTES
# ===========================================================================

# ---------------------------------------------------------------------------
# POST /create — Create Workspace
# ---------------------------------------------------------------------------
@router.post("/create")
async def create_workspace(
    req: CreateWorkspaceRequest,
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    ws = Workspace(
        name=req.name,
        user_id=user["id"],
    )
    # Add owner as member
    owner_member = WorkspaceMember(
        workspace_id=ws.id,
        user_id=user["id"],
        role="owner",
    )
    ws.members.append(owner_member)

    db.add(ws)
    await db.flush()

    # Invalidate workspace list cache
    await cache_delete(workspace_list_key(user["id"]))

    logger.info("Workspace created: %s by %s", ws.id, user["id"])
    return _ws_to_dict(ws)


# ---------------------------------------------------------------------------
# GET / — List Workspaces (PAGINATED)
# ---------------------------------------------------------------------------
@router.get("/")
async def list_workspaces(
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
):
    user_id = user["id"]
    offset = (page - 1) * per_page

    # Count total
    count_result = await db.execute(
        select(func.count(Workspace.id)).where(
            or_(
                Workspace.user_id == user_id,
                Workspace.id.in_(
                    select(WorkspaceMember.workspace_id).where(WorkspaceMember.user_id == user_id)
                ),
            )
        )
    )
    total = count_result.scalar() or 0

    result = await db.execute(
        select(Workspace)
        .options(selectinload(Workspace.members))
        .where(
            or_(
                Workspace.user_id == user_id,
                Workspace.members.any(WorkspaceMember.user_id == user_id),
            )
        )
        .order_by(Workspace.updated_at.desc())
        .offset(offset)
        .limit(per_page)
    )
    workspaces = result.scalars().unique().all()

    return {
        "workspaces": [_ws_to_dict(ws) for ws in workspaces],
        "pagination": {"page": page, "per_page": per_page, "total": total},
    }


# ---------------------------------------------------------------------------
# GET /{workspace_id} — Get Workspace Details
# ---------------------------------------------------------------------------
@router.get("/{workspace_id}")
async def get_workspace(
    workspace_id: str,
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    ws = await _get_workspace_or_404(workspace_id, db)
    if not _is_member(ws, user["id"]):
        raise HTTPException(status_code=403, detail="Not authorized")

    # Get document count
    doc_count_result = await db.execute(
        select(func.count(Document.id)).where(Document.workspace_id == workspace_id)
    )
    doc_count = doc_count_result.scalar() or 0

    return _ws_to_dict(ws, docs_count=doc_count)


# ---------------------------------------------------------------------------
# PUT /{workspace_id} — Update Workspace
# ---------------------------------------------------------------------------
@router.put("/{workspace_id}")
async def update_workspace(
    workspace_id: str,
    req: UpdateWorkspaceRequest,
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    ws = await _get_workspace_or_404(workspace_id, db)
    if not _is_owner(ws, user["id"]):
        raise HTTPException(status_code=403, detail="Only owner can update workspace")

    if req.name is not None:
        ws.name = req.name

    # Invalidate caches
    await cache_delete(workspace_key(workspace_id))
    await cache_delete(workspace_list_key(user["id"]))

    return _ws_to_dict(ws)


# ---------------------------------------------------------------------------
# DELETE /{workspace_id} — Delete Workspace (cascade)
# ---------------------------------------------------------------------------
@router.delete("/{workspace_id}")
async def delete_workspace(
    workspace_id: str,
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    ws = await _get_workspace_or_404(workspace_id, db)
    if not _is_owner(ws, user["id"]):
        raise HTTPException(status_code=403, detail="Only owner can delete workspace")

    # Delete vectors from Qdrant
    try:
        from app.services.vector_store import delete_by_workspace
        await delete_by_workspace(workspace_id)
    except Exception as e:
        logger.warning("Failed to delete Qdrant vectors: %s", e)

    # Delete workspace (cascades to members, documents, invitations, search_history)
    await db.delete(ws)

    # Invalidate all caches for this workspace
    await cache_delete_pattern(f"ws:{workspace_id}*")
    await cache_delete(workspace_list_key(user["id"]))

    logger.info("Workspace deleted: %s", workspace_id)
    return {"status": "deleted", "workspace_id": workspace_id}


# ---------------------------------------------------------------------------
# POST /{workspace_id}/invite — Send Invitation
# ---------------------------------------------------------------------------
@router.post("/{workspace_id}/invite")
async def invite_to_workspace(
    workspace_id: str,
    req: ShareWorkspaceRequest,
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    ws = await _get_workspace_or_404(workspace_id, db)
    if not _is_owner(ws, user["id"]):
        raise HTTPException(status_code=403, detail="Only owner can invite members")

    # Check if target user exists
    result = await db.execute(select(User).where(User.email == req.email))
    target_user = result.scalar_one_or_none()
    if not target_user:
        raise HTTPException(status_code=404, detail="User with this email not found")

    # Check if already a member
    if _is_member(ws, target_user.id):
        raise HTTPException(status_code=400, detail="User is already a member")

    # Check for existing pending invitation
    result = await db.execute(
        select(Invitation).where(
            and_(
                Invitation.workspace_id == workspace_id,
                Invitation.invitee_email == req.email,
                Invitation.status == "pending",
            )
        )
    )
    if result.scalar_one_or_none():
        raise HTTPException(status_code=400, detail="Invitation already pending")

    # Create invitation (expires in 7 days)
    invitation = Invitation(
        workspace_id=workspace_id,
        inviter_id=user["id"],
        invitee_email=req.email,
        role=req.role,
        status="pending",
        expires_at=datetime.now(timezone.utc) + timedelta(days=7),
    )
    db.add(invitation)
    await db.flush()

    logger.info("Invitation sent: %s → %s for workspace %s", user["id"], req.email, workspace_id)
    return {
        "status": "invited",
        "invitation_id": invitation.id,
        "invitee_email": req.email,
        "role": req.role,
        "expires_at": invitation.expires_at.isoformat(),
    }


# ---------------------------------------------------------------------------
# DELETE /{workspace_id}/members/{user_id} — Remove Member
# ---------------------------------------------------------------------------
@router.delete("/{workspace_id}/members/{member_user_id}")
async def remove_member(
    workspace_id: str,
    member_user_id: str,
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    ws = await _get_workspace_or_404(workspace_id, db)
    if not _is_owner(ws, user["id"]):
        raise HTTPException(status_code=403, detail="Only owner can remove members")

    if member_user_id == user["id"]:
        raise HTTPException(status_code=400, detail="Cannot remove yourself as owner")

    result = await db.execute(
        select(WorkspaceMember).where(
            and_(
                WorkspaceMember.workspace_id == workspace_id,
                WorkspaceMember.user_id == member_user_id,
            )
        )
    )
    member = result.scalar_one_or_none()
    if not member:
        raise HTTPException(status_code=404, detail="Member not found")

    await db.delete(member)

    await cache_delete(workspace_key(workspace_id))
    await cache_delete(workspace_list_key(member_user_id))

    return {"status": "removed", "user_id": member_user_id}


# ---------------------------------------------------------------------------
# PUT /{workspace_id}/members/{user_id}/role — Change Member Role
# ---------------------------------------------------------------------------
@router.put("/{workspace_id}/members/{member_user_id}/role")
async def update_member_role(
    workspace_id: str,
    member_user_id: str,
    req: UpdateMemberRoleRequest,
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    ws = await _get_workspace_or_404(workspace_id, db)
    if not _is_owner(ws, user["id"]):
        raise HTTPException(status_code=403, detail="Only owner can change roles")

    if req.role not in ("editor", "viewer"):
        raise HTTPException(status_code=400, detail="Role must be 'editor' or 'viewer'")

    result = await db.execute(
        select(WorkspaceMember).where(
            and_(
                WorkspaceMember.workspace_id == workspace_id,
                WorkspaceMember.user_id == member_user_id,
                WorkspaceMember.role != "owner",
            )
        )
    )
    member = result.scalar_one_or_none()
    if not member:
        raise HTTPException(status_code=404, detail="Member not found or cannot change owner role")

    member.role = req.role

    await cache_delete(workspace_key(workspace_id))

    return {"status": "updated", "user_id": member_user_id, "new_role": req.role}


# ---------------------------------------------------------------------------
# POST /upload — Upload Document to Workspace (with file size limit)
# ---------------------------------------------------------------------------
@router.post("/upload")
async def upload_document(
    workspace_id: str = None,
    file: UploadFile = File(...),
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    # P2 FIX: File size validation
    contents = await file.read()
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {MAX_UPLOAD_BYTES // (1024*1024)} MB.",
        )
    # Reset file position for processing
    await file.seek(0)

    if not workspace_id:
        # Auto-create workspace
        ws = Workspace(name="New Workspace", user_id=user["id"])
        owner_member = WorkspaceMember(
            workspace_id=ws.id, user_id=user["id"], role="owner"
        )
        ws.members.append(owner_member)
        db.add(ws)
        await db.flush()
        workspace_id = ws.id
    else:
        ws = await _get_workspace_or_404(workspace_id, db)
        if not _is_member(ws, user["id"]):
            raise HTTPException(status_code=403, detail="Not authorized")

        # Check edit permission
        role = _get_role(ws, user["id"])
        if role == "viewer":
            raise HTTPException(status_code=403, detail="Viewers cannot upload documents")

    doc_text = await DocumentProcessor.process_file(file)
    if not doc_text.strip():
        raise HTTPException(status_code=422, detail="Empty document")

    doc_id = str(uuid.uuid4())

    # Ingest chunks into Qdrant vector DB
    result = await ingest_document(doc_text, file.filename, workspace_id, doc_id)

    # Store document record in PostgreSQL
    doc = Document(
        id=doc_id,
        workspace_id=workspace_id,
        filename=file.filename,
        content_length=len(doc_text),
        chunks_indexed=result["chunks"],
    )
    db.add(doc)

    # Invalidate caches
    await cache_delete(workspace_docs_key(workspace_id))
    await cache_delete(workspace_list_key(user["id"]))

    return {"status": "uploaded", "workspace_id": workspace_id, "document_id": doc_id}


# ---------------------------------------------------------------------------
# GET /{workspace_id}/documents — List Documents (PAGINATED)
# ---------------------------------------------------------------------------
@router.get("/{workspace_id}/documents")
async def list_workspace_documents(
    workspace_id: str,
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
):
    ws = await _get_workspace_or_404(workspace_id, db)
    if not _is_member(ws, user["id"]):
        raise HTTPException(status_code=403, detail="Not authorized")

    offset = (page - 1) * per_page

    # Count total
    count_result = await db.execute(
        select(func.count(Document.id)).where(Document.workspace_id == workspace_id)
    )
    total = count_result.scalar() or 0

    result = await db.execute(
        select(Document)
        .where(Document.workspace_id == workspace_id)
        .order_by(Document.created_at.desc())
        .offset(offset)
        .limit(per_page)
    )
    docs = result.scalars().all()

    return {
        "documents": [
            {
                "id": d.id,
                "filename": d.filename,
                "content_length": d.content_length,
                "chunks_indexed": d.chunks_indexed,
                "created_at": d.created_at.isoformat() if d.created_at else None,
            }
            for d in docs
        ],
        "pagination": {"page": page, "per_page": per_page, "total": total},
    }


# ---------------------------------------------------------------------------
# DELETE /{workspace_id}/documents/{doc_id} — Delete a Single Document
# ---------------------------------------------------------------------------
@router.delete("/{workspace_id}/documents/{doc_id}")
async def delete_document(
    workspace_id: str,
    doc_id: str,
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    ws = await _get_workspace_or_404(workspace_id, db)
    role = _get_role(ws, user["id"])
    if role not in ("owner", "editor"):
        raise HTTPException(status_code=403, detail="Only owner/editor can delete documents")

    result = await db.execute(
        select(Document).where(
            and_(Document.id == doc_id, Document.workspace_id == workspace_id)
        )
    )
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    # Delete vectors from Qdrant for this specific document
    try:
        from app.services.vector_store import get_qdrant_client
        from qdrant_client.http.models import Filter, FieldCondition, MatchValue
        client = get_qdrant_client()
        from app.core.config import settings as cfg
        await client.delete(
            collection_name=cfg.QDRANT_COLLECTION,
            points_selector=Filter(
                must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
            ),
        )
    except Exception as e:
        logger.warning("Failed to delete Qdrant vectors for doc %s: %s", doc_id, e)

    await db.delete(doc)

    await cache_delete(workspace_docs_key(workspace_id))

    logger.info("Document deleted: %s from workspace %s", doc_id, workspace_id)
    return {"status": "deleted", "document_id": doc_id}


# ---------------------------------------------------------------------------
# POST /chat — Chat with Workspace Documents
# ---------------------------------------------------------------------------
@router.post("/chat")
async def chat_workspace(
    req: WorkspaceChatRequest,
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    ws = await _get_workspace_or_404(req.workspace_id, db)
    if not _is_member(ws, user["id"]):
        raise HTTPException(status_code=403, detail="Not authorized")

    # Log search history
    history = SearchHistory(
        user_id=user["id"],
        workspace_id=req.workspace_id,
        query=req.query,
        mode="rag",
    )
    db.add(history)

    return StreamingResponse(
        _sse_rag(req.query, req.workspace_id, top_k=6),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# GET /{workspace_id}/graph — Workspace Graph Data
# ---------------------------------------------------------------------------
@router.get("/{workspace_id}/graph")
async def get_workspace_graph(
    workspace_id: str,
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Returns graph data for workspace visualization.
    Nodes: workspace, documents, members, recent searches.
    Edges: document→workspace, member→workspace, search→workspace.
    """
    ws = await _get_workspace_or_404(workspace_id, db)
    if not _is_member(ws, user["id"]):
        raise HTTPException(status_code=403, detail="Not authorized")

    nodes = []
    edges = []

    # Workspace node (center)
    nodes.append({
        "id": ws.id,
        "type": "workspace",
        "label": ws.name,
        "data": {"created_at": ws.created_at.isoformat() if ws.created_at else None},
    })

    # Document nodes
    doc_result = await db.execute(
        select(Document).where(Document.workspace_id == workspace_id)
    )
    for doc in doc_result.scalars().all():
        nodes.append({
            "id": doc.id,
            "type": "document",
            "label": doc.filename,
            "data": {
                "chunks": doc.chunks_indexed,
                "size": doc.content_length,
                "created_at": doc.created_at.isoformat() if doc.created_at else None,
            },
        })
        edges.append({
            "source": doc.id,
            "target": ws.id,
            "type": "belongs_to",
            "label": "in workspace",
        })

    # Member nodes
    for member in ws.members:
        member_result = await db.execute(select(User).where(User.id == member.user_id))
        member_user = member_result.scalar_one_or_none()
        if member_user:
            nodes.append({
                "id": member.user_id,
                "type": "user",
                "label": member_user.name,
                "data": {"role": member.role, "email": member_user.email},
            })
            edges.append({
                "source": member.user_id,
                "target": ws.id,
                "type": "member_of",
                "label": member.role,
            })

    # Search history nodes (last 20)
    search_result = await db.execute(
        select(SearchHistory)
        .where(SearchHistory.workspace_id == workspace_id)
        .order_by(SearchHistory.created_at.desc())
        .limit(20)
    )
    for sh in search_result.scalars().all():
        search_node_id = f"search_{sh.id}"
        nodes.append({
            "id": search_node_id,
            "type": "search",
            "label": sh.query[:50] + ("..." if len(sh.query) > 50 else ""),
            "data": {
                "full_query": sh.query,
                "mode": sh.mode,
                "user_id": sh.user_id,
                "created_at": sh.created_at.isoformat() if sh.created_at else None,
            },
        })
        edges.append({
            "source": search_node_id,
            "target": ws.id,
            "type": "searched_in",
            "label": "search",
        })

    return {
        "workspace_id": workspace_id,
        "nodes": nodes,
        "edges": edges,
        "stats": {
            "documents": len([n for n in nodes if n["type"] == "document"]),
            "members": len([n for n in nodes if n["type"] == "user"]),
            "searches": len([n for n in nodes if n["type"] == "search"]),
        },
    }
