"""
Action plugin routes — REQUIRES AUTH.
GET  /actions/       → list available plugins
POST /actions/execute → execute a plugin with confirmation gate
"""
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Dict, Any

from app.agents.plugins import plugin_registry
from app.core.security import get_current_user

router = APIRouter()


class ActionRequest(BaseModel):
    plugin_name: str
    parameters: dict
    user_confirmed: bool = False


@router.get("/")
async def list_plugins(user: dict = Depends(get_current_user)):
    return {"plugins": [p.dict() for p in plugin_registry.plugins.values()]}


@router.post("/execute")
async def execute_plugin(
    req: ActionRequest,
    user: dict = Depends(get_current_user),
):
    # Privacy Gate: Require explicit user confirmation for actions
    if not req.user_confirmed:
        return {
            "status": "pending_confirmation",
            "message": f"I am about to send your request to {req.plugin_name}. Allow?",
            "payload_to_send": req.parameters,
        }

    try:
        result = await plugin_registry.execute(req.plugin_name, req.parameters)
        return {"status": "executed", "result": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
