import asyncio
import logging
from duckduckgo_search import DDGS
from bs4 import BeautifulSoup
import trafilatura
import httpx
from typing import List

logger = logging.getLogger(__name__)

import random

# ---------------------------------------------------------------------------
# User-Agent rotation for scraping resilience
# ---------------------------------------------------------------------------
_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4_1) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4.1 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64; rv:125.0) Gecko/20100101 Firefox/125.0",
]

def _next_ua() -> str:
    return random.choice(_USER_AGENTS)


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------
def _sync_ddg_search(query: str, max_results: int) -> List[dict]:
    """Synchronous DDG search — called via asyncio.to_thread."""
    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results):
            results.append(
                {"title": r.get("title"), "href": r.get("href"), "body": r.get("body")}
            )
    return results


async def search_web(query: str, max_results: int = 5) -> List[dict]:
    """Non-blocking web search using DuckDuckGo."""
    try:
        results = await asyncio.to_thread(_sync_ddg_search, query, max_results)
        logger.info("Search completed", extra={"query": query, "results": len(results)})
        return results
    except Exception as e:
        logger.warning("Search failed, returning empty", extra={"query": query, "error": str(e)})
        return []


# ---------------------------------------------------------------------------
# Content Extraction
# ---------------------------------------------------------------------------
async def fetch_and_extract_content(url: str) -> dict:
    """Async fetch + content extraction with trafilatura fallback to BS4."""
    timeout = httpx.Timeout(8.0, connect=4.0)
    headers = {"User-Agent": _next_ua()}

    async with httpx.AsyncClient(
        timeout=timeout, follow_redirects=True, headers=headers
    ) as client:
        try:
            response = await client.get(url)
            response.raise_for_status()
            html = response.text

            # Primary extractor
            extracted = trafilatura.extract(
                html, include_links=False, include_images=False, include_formatting=True
            )
            if extracted:
                return {"url": url, "content": extracted, "success": True}

            # Fallback
            soup = BeautifulSoup(html, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "aside", "header"]):
                tag.extract()
            text = soup.get_text(separator=" ", strip=True)
            return {"url": url, "content": text[:5000], "success": True}

        except httpx.TimeoutException:
            logger.warning("Scrape timeout", extra={"url": url})
            return {"url": url, "content": "", "success": False, "error": "timeout"}
        except Exception as e:
            logger.warning("Scrape failed", extra={"url": url, "error": str(e)})
            return {"url": url, "content": "", "success": False, "error": str(e)}


# ---------------------------------------------------------------------------
# Parallel Pipeline
# ---------------------------------------------------------------------------
async def process_search_queries(
    queries: List[str], max_results_per_query: int = 3
) -> List[dict]:
    """Execute multiple search queries and extract content in parallel."""
    all_urls: set = set()
    mapped_results: dict = {}

    # 1. Search Phase (parallel)
    search_tasks = [search_web(q, max_results_per_query) for q in queries]
    search_responses = await asyncio.gather(*search_tasks)

    for query_res in search_responses:
        for res in query_res:
            url = res.get("href")
            if url and url not in all_urls:
                all_urls.add(url)
                mapped_results[url] = {
                    "title": res["title"],
                    "snippet": res["body"],
                    "url": url,
                }

    if not all_urls:
        logger.warning("No URLs found from search phase")
        return []

    # 2. Extract Phase (parallel with concurrency limit)
    sem = asyncio.Semaphore(6)

    async def _bounded_extract(u: str):
        async with sem:
            return await fetch_and_extract_content(u)

    extract_tasks = [_bounded_extract(url) for url in all_urls]
    extract_responses = await asyncio.gather(*extract_tasks, return_exceptions=True)

    final_context = []
    for ext_res in extract_responses:
        if isinstance(ext_res, Exception) or not ext_res.get("success"):
            continue
        url = ext_res["url"]
        page_data = mapped_results[url]
        page_data["extracted_content"] = ext_res["content"]
        final_context.append(page_data)

    logger.info(
        "Search pipeline completed",
        extra={"queries": len(queries), "urls_found": len(all_urls), "extracted": len(final_context)},
    )
    return final_context
