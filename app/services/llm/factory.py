from typing import Dict
from app.core.config import settings
from app.services.llm.base import AsyncAIInferenceClient
from app.services.llm.openai import AsyncOpenAIClient
from app.services.llm.gemini import AsyncGeminiClient

import time
import logging

logger = logging.getLogger(__name__)

# Circuit Breaker State
_CIRCUIT_STATE = {
    "groq_failures": 0,
    "groq_last_failure_time": 0,
    "groq_circuit_open": False,
}

CIRCUIT_FAILURE_THRESHOLD = 5
CIRCUIT_RECOVERY_TIME = 60  # seconds


def record_llm_failure(provider: str):
    """Record an LLM failure for circuit breaker."""
    if provider == "groq":
        _CIRCUIT_STATE["groq_failures"] += 1
        _CIRCUIT_STATE["groq_last_failure_time"] = time.time()
        if _CIRCUIT_STATE["groq_failures"] >= CIRCUIT_FAILURE_THRESHOLD:
            if not _CIRCUIT_STATE["groq_circuit_open"]:
                logger.error("CIRCUIT BREAKER OPEN: Groq failed %d times. Falling back to Gemini.", CIRCUIT_FAILURE_THRESHOLD)
                _CIRCUIT_STATE["groq_circuit_open"] = True


def record_llm_success(provider: str):
    """Reset circuit breaker on success."""
    if provider == "groq":
        if _CIRCUIT_STATE["groq_circuit_open"]:
            logger.info("CIRCUIT BREAKER CLOSED: Groq recovered.")
        _CIRCUIT_STATE["groq_circuit_open"] = False
        _CIRCUIT_STATE["groq_failures"] = 0


def get_llm_client(provider: str = "groq") -> AsyncAIInferenceClient:
    """Returns a singleton AsyncAIInferenceClient based on the provider for maximum TCP connection reuse."""
    
    # Circuit Breaker Check
    if provider == "groq":
        if _CIRCUIT_STATE["groq_circuit_open"]:
            # Check if recovery time has elapsed to attempt half-open state
            if time.time() - _CIRCUIT_STATE["groq_last_failure_time"] > CIRCUIT_RECOVERY_TIME:
                logger.info("CIRCUIT BREAKER HALF-OPEN: Testing Groq again.")
                _CIRCUIT_STATE["groq_circuit_open"] = False
            else:
                logger.warning("Groq circuit is open. Auto-switching to Gemini fallback.")
                provider = "gemini"  # Fallback to gemini

    provider_key = provider.lower()
    
    # Connection Cache allows reuse of the internal httpx.AsyncClient pool natively
    if provider_key in _CLIENT_CACHE:
        return _CLIENT_CACHE[provider_key]
        
    if provider_key == "groq":
        if not settings.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY is not configured.")
        client = AsyncOpenAIClient(
            base_url="https://api.groq.com/openai/v1",
            api_key=settings.GROQ_API_KEY,
        )
        _CLIENT_CACHE[provider_key] = client
        return client
    elif provider_key == "gemini":
        if not settings.GOOGLE_API_KEY:
            if provider != "gemini": # It means we fell back to gemini but it's not configured
                 raise ValueError("Groq is failing and GOOGLE_API_KEY fallback is not configured.")
            raise ValueError("GOOGLE_API_KEY is not configured.")
        client = AsyncGeminiClient(
            api_key=settings.GOOGLE_API_KEY,
        )
        _CLIENT_CACHE[provider_key] = client
        return client
        
    raise ValueError(f"Unsupported LLM provider: {provider}")


async def close_all_clients():
    """Closes all cached LLM clients to release HTTP connections."""
    import logging
    logger = logging.getLogger(__name__)
    for name, client in _CLIENT_CACHE.items():
        try:
            await client.close()
            logger.info("Closed LLM client: %s", name)
        except Exception as e:
            logger.warning("Error closing LLM client %s: %s", name, e)
    _CLIENT_CACHE.clear()
