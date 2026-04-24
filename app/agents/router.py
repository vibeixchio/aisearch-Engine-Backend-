import logging
from langchain_groq import ChatGroq
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from app.core.config import settings
from app.models.routing import IntentClassification

logger = logging.getLogger(__name__)

ROUTER_PROMPT = """You are the central intelligence router for a highly advanced search product.
Your job is to read the user's query and decide the absolute best way to handle it.

Modes:
- direct_answer: Simple conversational follow-up, greeting, or common knowledge.
- quick_search: Recent events, specific facts, quick summaries, current news. (Use 1-3 search queries).
- deep_research: Complex synthesis, comparisons over time, academic dives, or "reports".
- file_query: Explicitly asking about an uploaded document or workspace.
- agent: Multi-step tasks that require planning and taking actions (e.g., "build X", "research Y and then do Z").
- action: Wants to *do* something (book a flight, schedule).

Generate exactly the JSON output required by the schema."""


def get_router_model():
    if not settings.GROQ_API_KEY:
        logger.warning("GROQ_API_KEY not set — using fallback router")
        return None

    return ChatGroq(
        temperature=0,
        model_name=settings.GROQ_ROUTER_MODEL,
        groq_api_key=settings.GROQ_API_KEY,
    ).with_structured_output(IntentClassification)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
async def _invoke_router(llm, prompt: str) -> IntentClassification:
    return await llm.ainvoke(prompt)


async def route_query(query: str, chat_history: list = None) -> IntentClassification:
    """Route a query to the appropriate processing mode."""
    llm = get_router_model()
    if not llm:
        return IntentClassification(
            mode="quick_search",
            search_queries=[query],
            reasoning="Fallback — no API key configured",
        )

    prompt = f"{ROUTER_PROMPT}\n\nUser Query: {query}"
    if chat_history:
        prompt += f"\n\nRecent History: {chat_history[-3:]}"

    try:
        result = await _invoke_router(llm, prompt)
        logger.info("Routed query", extra={"mode": result.mode, "query": query[:80]})
        return result
    except Exception as e:
        logger.exception("Router failed after retries", extra={"query": query[:80]})
        return IntentClassification(
            mode="quick_search",
            search_queries=[query],
            reasoning=f"Fallback due to error: {str(e)[:100]}",
        )
