import logging
import json
from app.services.llm.factory import get_llm_client
from app.core.config import settings

logger = logging.getLogger(__name__)

REWRITE_PROMPT = """You are a query rewriting engine for an Indian AI Search Engine.
Your task is to take the user's raw query and output a clean, highly optimized search query in English.

Steps:
1. Translate from Hindi/Hinglish to English if needed.
2. Correct any spelling or grammar mistakes.
3. Extract core entities and intent.
4. If the query is already perfect English, just return it.

Return ONLY a JSON object in this exact format, with NO markdown formatting, NO backticks, and NO other text:
{"rewritten_query": "The clean english query"}

Raw Query: {query}
"""

async def rewrite_query(query: str) -> str:
    """Rewrites a query for better search results (Hinglish translation, spellcheck)."""
    try:
        client = get_llm_client("groq")
        response = await client.chat(
            model=settings.GROQ_ROUTING_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful JSON-only assistant."},
                {"role": "user", "content": REWRITE_PROMPT.format(query=query)}
            ],
            temperature=0.0,
            max_tokens=100
        )
        
        content = response.content.strip()
        if content.startswith("```json"):
            content = content[7:-3].strip()
        elif content.startswith("```"):
            content = content[3:-3].strip()
            
        data = json.loads(content)
        rewritten = data.get("rewritten_query", query)
        
        if rewritten.lower() != query.lower():
            logger.info("Query rewritten: '%s' -> '%s'", query, rewritten)
            
        return rewritten
    except Exception as e:
        logger.warning("Query rewriting failed for '%s': %s", query, e)
        return query  # Fallback to original
