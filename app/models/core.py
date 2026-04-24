from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime

class UserBase(BaseModel):
    id: str
    is_anonymous: bool = False
    cloud_enhanced: bool = True # Privacy Toggle
    created_at: datetime = Field(default_factory=datetime.utcnow)

class WorkspaceMember(BaseModel):
    user_id: str
    role: str = "viewer" # "owner", "editor", "viewer"

class Workspace(BaseModel):
    id: str
    user_id: str
    name: str = "Default Workspace"
    file_ids: List[str] = []
    members: List[WorkspaceMember] = []
    created_at: datetime = Field(default_factory=datetime.utcnow)

class SearchQuery(BaseModel):
    query: str
    mode: str = "general" # general, deep, coding, academic
    session_id: Optional[str] = None
    
class SearchResult(BaseModel):
    query: str
    answer: str
    sources: List[dict]
    processing_time_ms: int
