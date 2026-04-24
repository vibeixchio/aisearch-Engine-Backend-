import uuid
import logging
from typing import Dict, Any

from app.models.deep_research import ResearchState
from app.agents.deep_research import build_research_graph
from app.db.redis_client import redis_client

logger = logging.getLogger(__name__)


async def run_deep_research(job_id: str, query: str):
    """Execute a deep research job via LangGraph and track state in Redis."""
    try:
        state: ResearchState = {
            "original_query": query,
            "sub_queries": [],
            "executed_queries": [],
            "gathered_context": [],
            "research_summary": "Starting research...",
            "final_report": "",
            "iteration_count": 0,
            "max_iterations": 3,
        }

        await redis_client.redis_pool.hset(
            f"job:{job_id}", mapping={"status": "running", "step": "planning"}
        )

        graph = build_research_graph()

        # Stream events — collect the final state from the last emitted output
        final_state = state
        async for output in graph.astream(state):
            for node_name, node_state in output.items():
                logger.info("Deep research node finished", extra={"node": node_name, "job_id": job_id})
                final_state = {**final_state, **node_state}
                await redis_client.redis_pool.hset(
                    f"job:{job_id}",
                    mapping={
                        "step": node_name,
                        "summary": str(node_state.get("research_summary", "")),
                    },
                )

        report = final_state.get("final_report", "Research failed to compile.")
        sources_count = len(final_state.get("gathered_context", []))

        await redis_client.redis_pool.hset(
            f"job:{job_id}",
            mapping={
                "status": "completed",
                "report": report,
                "sources_count": str(sources_count),
            },
        )
        await redis_client.redis_pool.expire(f"job:{job_id}", 86400)
        logger.info("Deep research completed", extra={"job_id": job_id, "sources": sources_count})

    except Exception as e:
        logger.exception("Research job failed", extra={"job_id": job_id})
        if redis_client.redis_pool:
            await redis_client.redis_pool.hset(
                f"job:{job_id}", mapping={"status": "failed", "error": str(e)}
            )


async def create_research_job(query: str) -> str:
    job_id = str(uuid.uuid4())
    await redis_client.redis_pool.hset(
        f"job:{job_id}", mapping={"status": "pending", "query": query}
    )
    return job_id


async def get_job_status(job_id: str) -> Dict[str, Any]:
    data = await redis_client.redis_pool.hgetall(f"job:{job_id}")
    return data
