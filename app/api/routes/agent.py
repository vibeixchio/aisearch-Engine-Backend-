"""
Agent API route — requires auth.
POST /api/v1/agent/run → SSE stream of agent thought/action events
GET  /api/v1/agent/tools → list available tools
"""
import json
import logging
import uuid

from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional

from app.agents.controller import run_agent
from app.agents.tool_registry import tool_registry
from app.core.config import settings
from app.core.security import get_current_user

logger = logging.getLogger(__name__)
router = APIRouter()


class AgentRunRequest(BaseModel):
    goal: str
    session_id: Optional[str] = None
    workspace_id: Optional[str] = None
    max_iterations: Optional[int] = None


@router.get("/tools")
async def list_tools(user: dict = Depends(get_current_user)):
    """List all available agent tools."""
    return {"tools": tool_registry.list_schemas()}


@router.post("/run")
async def run_agent_endpoint(
    req: AgentRunRequest,
    user: dict = Depends(get_current_user),
):
    """
    Run the AI agent on a goal. Returns a Server-Sent Event stream.

    Event types:
    - `plan`         → structured list of steps the agent will take
    - `step_start`   → beginning of a step
    - `tool_call`    → exact tool name + input being dispatched
    - `observation`  → raw tool result
    - `review`       → critic assessment (approved, reason, next_action)
    - `done`         → final synthesized answer + stats
    - `error`        → fatal error
    """
    if not settings.GROQ_API_KEY:
        raise HTTPException(status_code=503, detail="GROQ_API_KEY not configured")

    if not req.goal.strip():
        raise HTTPException(status_code=400, detail="Goal cannot be empty")

    session_id = req.session_id or f"sess_{uuid.uuid4().hex[:8]}"

    return StreamingResponse(
        run_agent(
            goal=req.goal,
            session_id=session_id,
            workspace_id=req.workspace_id,
            max_iterations=req.max_iterations,
        ),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "X-Session-ID": session_id},
    )
