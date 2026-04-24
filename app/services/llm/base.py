import abc
import json
import httpx
from typing import List, Dict, Optional, Any, AsyncGenerator

class AsyncAIInferenceClient(abc.ABC):
    def __init__(self, base_url: str, api_key: Optional[str] = None, timeout: int = 60):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        
        # Instantiate a robust persistent connection pool natively
        self._client = httpx.AsyncClient(timeout=self.timeout)

    @abc.abstractmethod
    async def chat(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        extra_payload: Optional[Dict[str, Any]] = None,
    ) -> Any | AsyncGenerator[str, None]:
        pass

    def _handle_error(self, status_code: int, response_text: str):
        if status_code >= 400:
            try:
                err = json.loads(response_text)
            except json.JSONDecodeError:
                err = response_text
            raise Exception(f"API Error {status_code}: {err}")

    async def close(self):
        """Clean shutdown of the HTTP connection pool hooks"""
        await self._client.aclose()
