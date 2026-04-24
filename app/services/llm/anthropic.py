import json
from typing import List, Dict, Optional, Any, AsyncGenerator
from app.services.llm.base import AsyncAIInferenceClient

class AsyncAnthropicClient(AsyncAIInferenceClient):
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.anthropic.com/v1",
        anthropic_version: str = "2023-06-01",
        timeout: int = 60,
    ):
        super().__init__(base_url, api_key, timeout)
        self.headers = {
            "x-api-key": api_key,
            "anthropic-version": anthropic_version,
            "Content-Type": "application/json",
        }

    async def chat(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = 1024,
        stream: bool = False,
        system: Optional[str] = None,
        extra_payload: Optional[Dict[str, Any]] = None,
    ) -> Any | AsyncGenerator[str, None]:
        url = f"{self.base_url}/messages"
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
        }
        if system:
            payload["system"] = system
        if extra_payload:
            payload.update(extra_payload)

        if stream:
            return self._stream_request(url, payload)
            
        response = await self._client.post(url, headers=self.headers, json=payload)
        self._handle_error(response.status_code, response.text)
        return response.json()

    async def _stream_request(self, url: str, payload: Dict[str, Any]) -> AsyncGenerator[str, None]:
        async with self._client.stream("POST", url, headers=self.headers, json=payload) as response:
            if response.status_code >= 400:
                await response.aread()
                self._handle_error(response.status_code, response.text)
                
            async for line in response.aiter_lines():
                if not line:
                    continue
                if line.startswith("data:"):
                    data = line[5:].strip()
                    try:
                        chunk = json.loads(data)
                        if chunk.get("type") == "content_block_delta":
                            delta = chunk.get("delta", {}).get("text", "")
                            if delta:
                                yield delta
                    except json.JSONDecodeError:
                        continue
