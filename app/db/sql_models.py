"""
SQLAlchemy ORM models for PostgreSQL (Supabase).
All relational data: users, workspaces, members, documents, invitations, search history.
"""
import uuid
from datetime import datetime, timezone
from sqlalchemy import (
    Column, String, Boolean, DateTime, Integer, Float,
    ForeignKey, Text, Index, Enum as SAEnum
)
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID

from app.db.database import Base


def utcnow():
    return datetime.now(timezone.utc)


def gen_uuid():
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# User
# ---------------------------------------------------------------------------
class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True, default=lambda: f"usr_{uuid.uuid4().hex[:12]}")
    email = Column(String, unique=True, nullable=False, index=True)
    name = Column(String, nullable=False)
    hashed_password = Column(String, nullable=False)
    is_verified = Column(Boolean, default=False)
    verification_code = Column(String, nullable=True)
    verification_expires = Column(DateTime(timezone=True), nullable=True)
    cloud_enhanced = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), default=utcnow)
    updated_at = Column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    # Relationships
    owned_workspaces = relationship("Workspace", back_populates="owner", cascade="all, delete-orphan")
    memberships = relationship("WorkspaceMember", back_populates="user", cascade="all, delete-orphan")
    sent_invitations = relationship("Invitation", foreign_keys="Invitation.inviter_id", back_populates="inviter")
    search_history = relationship("SearchHistory", back_populates="user", cascade="all, delete-orphan")


# ---------------------------------------------------------------------------
# Workspace
# ---------------------------------------------------------------------------
class Workspace(Base):
    __tablename__ = "workspaces"

    id = Column(String, primary_key=True, default=lambda: f"ws_{uuid.uuid4().hex[:8]}")
    name = Column(String, nullable=False, default="New Workspace")
    user_id = Column(String, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    created_at = Column(DateTime(timezone=True), default=utcnow)
    updated_at = Column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    # Relationships
    owner = relationship("User", back_populates="owned_workspaces")
    members = relationship("WorkspaceMember", back_populates="workspace", cascade="all, delete-orphan")
    documents = relationship("Document", back_populates="workspace", cascade="all, delete-orphan")
    search_history = relationship("SearchHistory", back_populates="workspace", cascade="all, delete-orphan")
    invitations = relationship("Invitation", back_populates="workspace", cascade="all, delete-orphan")


# ---------------------------------------------------------------------------
# Workspace Member (many-to-many: user ↔ workspace with role)
# ---------------------------------------------------------------------------
class WorkspaceMember(Base):
    __tablename__ = "workspace_members"

    id = Column(String, primary_key=True, default=gen_uuid)
    workspace_id = Column(String, ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False)
    user_id = Column(String, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    role = Column(String, nullable=False, default="viewer")  # owner, editor, viewer
    joined_at = Column(DateTime(timezone=True), default=utcnow)

    # Relationships
    workspace = relationship("Workspace", back_populates="members")
    user = relationship("User", back_populates="memberships")

    __table_args__ = (
        Index("ix_ws_member_unique", "workspace_id", "user_id", unique=True),
    )


# ---------------------------------------------------------------------------
# Document
# ---------------------------------------------------------------------------
class Document(Base):
    __tablename__ = "documents"

    id = Column(String, primary_key=True, default=gen_uuid)
    workspace_id = Column(String, ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False, index=True)
    filename = Column(String, nullable=False)
    content_length = Column(Integer, default=0)
    chunks_indexed = Column(Integer, default=0)
    created_at = Column(DateTime(timezone=True), default=utcnow)

    # Relationships
    workspace = relationship("Workspace", back_populates="documents")


# ---------------------------------------------------------------------------
# Invitation
# ---------------------------------------------------------------------------
class Invitation(Base):
    __tablename__ = "invitations"

    id = Column(String, primary_key=True, default=gen_uuid)
    workspace_id = Column(String, ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False, index=True)
    inviter_id = Column(String, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    invitee_email = Column(String, nullable=False, index=True)
    role = Column(String, nullable=False, default="viewer")
    status = Column(String, nullable=False, default="pending")  # pending, accepted, rejected, expired
    created_at = Column(DateTime(timezone=True), default=utcnow)
    expires_at = Column(DateTime(timezone=True), nullable=True)

    # Relationships
    workspace = relationship("Workspace", back_populates="invitations")
    inviter = relationship("User", foreign_keys=[inviter_id], back_populates="sent_invitations")


# ---------------------------------------------------------------------------
# Search History (for workspace graph)
# ---------------------------------------------------------------------------
class SearchHistory(Base):
    __tablename__ = "search_history"

    id = Column(String, primary_key=True, default=gen_uuid)
    user_id = Column(String, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    workspace_id = Column(String, ForeignKey("workspaces.id", ondelete="SET NULL"), nullable=True, index=True)
    query = Column(Text, nullable=False)
    mode = Column(String, default="instant")  # instant, deep, rag, agent
    response_preview = Column(Text, nullable=True)
    processing_time_ms = Column(Integer, nullable=True)
    feedback_rating = Column(Integer, nullable=True)  # 1 for upvote, -1 for downvote, null for none
    created_at = Column(DateTime(timezone=True), default=utcnow)

    # Relationships
    user = relationship("User", back_populates="search_history")
    workspace = relationship("Workspace", back_populates="search_history")
