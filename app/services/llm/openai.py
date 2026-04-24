import json
from typing import List, Dict, Optional, Any, AsyncGenerator
from app.services.llm.base import AsyncAIInferenceClient

class AsyncOpenAIClient(AsyncAIInferenceClient):
    def __init__(
        self,
        base_url: str = "https://api.openai.com/v1",
        api_key: Optional[str] = None,
        timeout: int = 60,
    ):
        super().__init__(base_url, api_key, timeout)
        self.headers = {"Content-Type": "application/json"}
        if self.api_key:
            self.headers["Authorization"] = f"Bearer {self.api_key}"

    async def chat(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        extra_payload: Optional[Dict[str, Any]] = None,
    ) -> Any | AsyncGenerator[str, None]:
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": stream,
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens
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
                    # Tolerate older streaming architectures correctly
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        choices = chunk.get("choices", [])
                        if choices:
                            delta = choices[0].get("delta", {}).get("content", "")
                            if delta:
                                yield delta
                    except json.JSONDecodeError:
                        continue
