"""
Full RAG query pipeline.
Flow: embed query → hybrid retrieval (vector + keyword) → RRF fusion → rerank → prompt → stream LLM.
"""
import asyncio
import json
import logging
from typing import AsyncGenerator, List, Dict, Any, Optional, Tuple

from app.core.config import settings
from app.services.llm import get_llm_client
from app.services.embedder import embed_query
from app.services.vector_store import vector_search, keyword_search
from app.services.reranker import rerank

logger = logging.getLogger(__name__)

RAG_PROMPT = """You are a precise AI assistant. Answer the user's question using ONLY the provided document context.

Rules:
1. Answer accurately and comprehensively using the context below.
2. Use inline citations in the format [N] for each factual claim, where N is the Source ID.
3. If the answer cannot be found in the context, clearly say: "I couldn't find this in the provided documents."
4. Format your response using clean Markdown.

Context:
{context}

Question: {query}

Answer:"""


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------
def _reciprocal_rank_fusion(
    vector_results: List[Dict],
    keyword_results: List[Dict],
    k: int = 60,
) -> List[Dict]:
    """
    Merge vector and keyword results using Reciprocal Rank Fusion.
    Higher RRF score = better combined rank.
    """
    scores: Dict[str, float] = {}
    items: Dict[str, Dict] = {}

    def _key(item: Dict) -> str:
        return f"{item.get('doc_id', '')}_{item.get('chunk_index', 0)}"

    for rank, item in enumerate(vector_results):
        key = _key(item)
        scores[key] = scores.get(key, 0) + 1 / (k + rank + 1)
        items[key] = item

    for rank, item in enumerate(keyword_results):
        key = _key(item)
        scores[key] = scores.get(key, 0) + 1 / (k + rank + 1)
        if key not in items:
            items[key] = item

    sorted_keys = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    merged = []
    for key in sorted_keys:
        item = items[key].copy()
        item["rrf_score"] = scores[key]
        merged.append(item)
    return merged


# ---------------------------------------------------------------------------
# Context formatter
# ---------------------------------------------------------------------------
def _format_context(chunks: List[Dict]) -> Tuple[str, List[Dict]]:
    """Format top chunks into context string and sources list."""
    context_parts = []
    sources = []
    for idx, chunk in enumerate(chunks, 1):
        text = chunk.get("text", "")[:2000]
        source = chunk.get("source", "Document")
        context_parts.append(f"[{idx}] Source: {source}\n{text}")
        sources.append({
            "id": idx,
            "title": source,
            "doc_id": chunk.get("doc_id"),
            "chunk_index": chunk.get("chunk_index"),
            "rerank_score": chunk.get("rerank_score"),
        })
    return "\n\n---\n\n".join(context_parts), sources


# ---------------------------------------------------------------------------
# Main RAG query function
# ---------------------------------------------------------------------------
async def rag_query(
    query: str,
    workspace_id: Optional[str] = None,
    top_k: int = None,
) -> Tuple[AsyncGenerator, List[Dict]]:
    """
    Full RAG pipeline.

    Returns:
        (stream_generator, sources_list)
    """
    top_k = top_k or settings.RAG_TOP_K
    fetch_k = top_k * 3  # fetch more, then rerank down to top_k

    if not settings.GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY must be set for RAG synthesis.")

    # 1. Embed query (thread executor)
    query_vector = await asyncio.to_thread(embed_query, query)

    # 2. Hybrid retrieval (parallel)
    vec_task = vector_search(query_vector, workspace_id=workspace_id, top_k=fetch_k)
    kw_task = keyword_search(query, workspace_id=workspace_id, top_k=fetch_k)
    vector_results, keyword_results = await asyncio.gather(vec_task, kw_task)

    logger.info(
        "Hybrid retrieval: vector=%d keyword=%d workspace=%s",
        len(vector_results), len(keyword_results), workspace_id,
    )

    # 3. Reciprocal Rank Fusion
    fused = _reciprocal_rank_fusion(vector_results, keyword_results)

    # 4. Re-rank with cross-encoder (thread executor)
    if fused:
        reranked = await asyncio.to_thread(rerank, query, fused, top_k)
    else:
        reranked = []

    logger.info("After reranking: %d chunks", len(reranked))

    # 5. Format context
    context_str, sources = _format_context(reranked)

    if not sources:
        async def _empty_gen():
            yield "I couldn't find relevant information in the documents. Please upload documents first or try a different query."
        return _empty_gen(), []

    # 6. Stream LLM
    llm_client = get_llm_client("groq")
    prompt = RAG_PROMPT.format(context=context_str, query=query)
    
    generator = await llm_client.chat(
        model=settings.GROQ_SYNTHESIS_MODEL,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.1,
        stream=True,
    )

    logger.info("RAG stream started: query=%s sources=%d", query[:80], len(sources))
    return generator, sources
