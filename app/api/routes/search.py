"""
Search routes — instant AI search with auth, caching, and search history tracking.
"""
import json
import logging
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.routing import IntentClassification
from app.agents.router import route_query
from app.agents.synthesis import synthesize_answer
from app.services.cache_layer import (
    cache_get_or_fetch, cache_set, search_cache_key,
)
from app.core.config import settings
from app.core.security import get_current_user
from app.db.database import get_db
from app.db.sql_models import SearchHistory

logger = logging.getLogger(__name__)
router = APIRouter()


class SearchRequest(BaseModel):
    query: str
    session_id: Optional[str] = None


async def generate_sse_response(query: str, route: IntentClassification, history_id: str):
    """SSE generator that streams sources → chunks → done."""
    import time
    start_time = time.time()
    
    if route.mode in ("direct_answer", "action", "file_query"):
        search_queries = [query]
    elif route.mode == "quick_search":
        search_queries = route.search_queries if route.search_queries else [query]
    else:
        search_queries = [query]

    try:
        generator, sources = await synthesize_answer(query, search_queries)

        yield f'data: {json.dumps({"type": "sources", "data": sources})}\n\n'

        full_response = []
        first_token_time = None
        
        async for chunk in generator:
            if first_token_time is None:
                first_token_time = time.time()
            full_response.append(chunk)
            yield f'data: {json.dumps({"type": "chunk", "text": chunk})}\n\n'

        end_time = time.time()
        ttft_ms = int((first_token_time - start_time) * 1000) if first_token_time else 0
        total_time_ms = int((end_time - start_time) * 1000)
        
        logger.info("Stream metrics for %s: TTFT=%dms, Total=%dms, Tokens~=%d", 
                    history_id, ttft_ms, total_time_ms, len(full_response))

        yield f'data: {json.dumps({"type": "done", "history_id": history_id})}\n\n'

        # Cache the full response
        try:
            from app.services.semantic_cache import set_semantic_cache
            answer_data = {"answer": "".join(full_response), "sources": sources}
            
            # Save to exact-match Redis cache
            await cache_set(
                search_cache_key(query),
                json.dumps(answer_data),
                ttl=settings.REDIS_SEARCH_TTL,
            )
            # Save to semantic vector cache (Qdrant)
            import asyncio
            asyncio.create_task(set_semantic_cache(query, answer_data))
        except Exception as e:
            logger.warning("Failed to save caches: %s", e)

    except Exception as e:
        logger.exception("SSE generation error", extra={"query": query[:80]})
        yield f'data: {json.dumps({"type": "error", "message": str(e)})}\n\n'


@router.post("/instant")
async def instant_search(
    req: SearchRequest,
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Real-time AI search with streaming LLM answers and inline citations.
    Requires authentication. Caches results in Redis.

    Returns a Server-Sent Event stream with three event types:
    - `sources`: Array of source objects (id, title, url, snippet)
    - `chunk`: Incremental text chunk from the LLM
    - `done`: Stream completion signal
    """
    if not settings.GROQ_API_KEY:
        raise HTTPException(status_code=503, detail="LLM API key not configured")

    # Check Semantic Cache first
    from app.services.semantic_cache import get_semantic_cache
    semantic_cached = await get_semantic_cache(req.query)
    if semantic_cached:
        return {
            "answer": semantic_cached.get("answer"),
            "cached": True,
            "semantic_cache": True,
            "sources": semantic_cached.get("sources", [])
        }

    # Fallback to Exact Match Redis cache
    from app.services.cache_layer import cache_get
    cached = await cache_get(search_cache_key(req.query))
    if cached:
        try:
            cached_data = json.loads(cached)
            return {"answer": cached_data.get("answer", cached), "cached": True, "sources": cached_data.get("sources", [])}
        except (json.JSONDecodeError, TypeError):
            return {"answer": cached, "cached": True}

    # Log search history with original query
    history = SearchHistory(
        user_id=user["id"],
        query=req.query,
        mode="instant",
    )
    db.add(history)
    await db.flush()  # get the history.id

    # Rewrite query for better search results
    from app.services.query_rewriter import rewrite_query
    rewritten_q = await rewrite_query(req.query)

    # Route the intent using the rewritten query
    route: IntentClassification = await route_query(rewritten_q)

    if route.mode == "deep_research":
        return {
            "redirect": True,
            "message": "This query requires deep research. Use POST /api/v1/search/deep/ instead.",
            "route": route.model_dump(),
        }

    return StreamingResponse(
        generate_sse_response(req.query, route, history.id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


class FeedbackRequest(BaseModel):
    history_id: str
    rating: int  # 1 for upvote, -1 for downvote

@router.post("/feedback")
async def submit_feedback(
    req: FeedbackRequest,
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    from sqlalchemy import select
    result = await db.execute(select(SearchHistory).where(SearchHistory.id == req.history_id))
    history = result.scalar_one_or_none()
    
    if not history:
        raise HTTPException(status_code=404, detail="Search history not found")
        
    if history.user_id != user["id"]:
        raise HTTPException(status_code=403, detail="Not authorized")
        
    if req.rating not in (1, -1):
        raise HTTPException(status_code=400, detail="Rating must be 1 or -1")
        
    history.feedback_rating = req.rating
    await db.commit()
    
    return {"status": "success", "message": "Feedback recorded successfully."}
