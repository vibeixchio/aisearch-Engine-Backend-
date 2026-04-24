"""
Deep search routes — long-running research jobs with auth.
"""
from fastapi import APIRouter, Depends, BackgroundTasks, HTTPException
from pydantic import BaseModel
from typing import Dict, Any

from app.services.job_runner import create_research_job, run_deep_research, get_job_status
from app.core.security import get_current_user

router = APIRouter()


class DeepSearchRequest(BaseModel):
    query: str


class JobResponse(BaseModel):
    job_id: str
    status: str


@router.post("/", response_model=JobResponse)
async def start_deep_research(
    req: DeepSearchRequest,
    background_tasks: BackgroundTasks,
    user: dict = Depends(get_current_user),
):
    job_id = await create_research_job(req.query)
    background_tasks.add_task(run_deep_research, job_id, req.query)
    return JobResponse(job_id=job_id, status="pending")


@router.get("/{job_id}")
async def check_job_status(
    job_id: str,
    user: dict = Depends(get_current_user),
) -> Dict[str, Any]:
    status = await get_job_status(job_id)
    if not status:
        raise HTTPException(status_code=404, detail="Job not found")
    return status
