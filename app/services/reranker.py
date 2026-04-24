"""
Cross-encoder re-ranker service.
Lazy-loaded singleton — scores (query, passage) pairs and re-orders results.
"""
import logging
from typing import List, Dict, Any

from app.core.config import settings

logger = logging.getLogger(__name__)

_reranker = None


def get_reranker():
    global _reranker
    if _reranker is None:
        logger.info("Loading reranker model: %s", settings.RERANKER_MODEL)
        from sentence_transformers import CrossEncoder
        _reranker = CrossEncoder(settings.RERANKER_MODEL, max_length=512)
        logger.info("Reranker model loaded")
    return _reranker


def rerank(query: str, candidates: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
    """
    Re-rank candidate chunks by cross-encoder relevance score.

    Args:
        query:      The user query.
        candidates: List of dicts, each must have 'text' key.
        top_k:      How many top results to return.

    Returns:
        Top-K candidates sorted by descending score, with 'rerank_score' added.
    """
    if not candidates:
        return []

    model = get_reranker()
    pairs = [(query, c.get("text", "")) for c in candidates]
    scores = model.predict(pairs)

    for candidate, score in zip(candidates, scores):
        candidate["rerank_score"] = float(score)

    ranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
    return ranked[:top_k]
