import json
import logging
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from app.core.config import settings
from app.agents.search_agent import process_search_queries

logger = logging.getLogger(__name__)

SYNTHESIS_PROMPT = """You are a highly intelligent, precise AI Search Engine.
Your goal is to answer the user's query comprehensively using ONLY the provided context.

Rules:
1. You MUST synthesize an accurate and highly readable answer.
2. You MUST use inline citations for every factual claim. Citations should be in the format [ID] where ID is the source ID provided in the context.
3. If the answer cannot be found in the context, clearly state that you do not have enough information rather than hallucinating.
4. Format your output in clean Markdown.

Context Materials:
{context}

User Query: {query}

Answer:"""

NO_SOURCES_RESPONSE = """I wasn't able to find relevant sources for your query at this time.

**Possible reasons:**
- The topic may be too niche for real-time search
- There may be temporary connectivity issues

Please try rephrasing your query or try again in a moment."""


def get_synthesis_model():
    if not settings.GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY must be set for synthesis.")

    return ChatGroq(
        temperature=0.2,
        model_name=settings.GROQ_SYNTHESIS_MODEL,
        groq_api_key=settings.GROQ_API_KEY,
        streaming=True,
    )


def format_context(extracted_data: list) -> str:
    formatted = []
    for idx, data in enumerate(extracted_data, 1):
        title = data.get("title", "Unknown Title")
        content = data.get("extracted_content", data.get("snippet", ""))[:3000]
        formatted.append(
            f"Source ID: [{idx}]\nTitle: {title}\nContent snippet:\n{content}\n"
        )
    return "\n---\n".join(formatted)


@retry(
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=0.5, min=0.5, max=3),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
async def _fetch_sources(search_queries: list[str]):
    return await process_search_queries(search_queries)


async def synthesize_answer(query: str, search_queries: list[str]):
    """Fetch sources and stream a synthesized answer with citations."""

    # 1. Fetch data with retry
    try:
        extracted_data = await _fetch_sources(search_queries)
    except Exception as e:
        logger.exception("Source fetching failed after retries")
        extracted_data = []

    # 2. Graceful fallback if no sources
    if not extracted_data:
        logger.warning("No sources extracted", extra={"query": query[:80]})

        async def _empty_gen():
            yield NO_SOURCES_RESPONSE

        return _empty_gen(), []

    # 3. Build source map
    sources_to_return = []
    for idx, data in enumerate(extracted_data, 1):
        sources_to_return.append(
            {
                "id": idx,
                "title": data.get("title"),
                "url": data.get("url"),
                "snippet": data.get("snippet"),
            }
        )

    # 4. Stream synthesis
    context_str = format_context(extracted_data)
    prompt = ChatPromptTemplate.from_template(SYNTHESIS_PROMPT)
    model = get_synthesis_model()
    chain = prompt | model | StrOutputParser()

    generator = chain.astream({"context": context_str, "query": query})
    logger.info("Synthesis started", extra={"query": query[:80], "sources": len(sources_to_return)})

    return generator, sources_to_return
