import json
from typing import List, Dict, Optional, Any, AsyncGenerator
from app.services.llm.base import AsyncAIInferenceClient

class AsyncGeminiClient(AsyncAIInferenceClient):
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://generativelanguage.googleapis.com/v1beta",
        timeout: int = 60,
    ):
        super().__init__(base_url, api_key, timeout)
        self.headers = {
            "x-goog-api-key": api_key,
            "Content-Type": "application/json"
        }

    async def chat(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        system_instruction: Optional[str] = None,
        extra_payload: Optional[Dict[str, Any]] = None,
    ) -> Any | AsyncGenerator[str, None]:
        endpoint = "streamGenerateContent" if stream else "generateContent"
        url = f"{self.base_url}/models/{model}:{endpoint}"
        
        contents = []
        for msg in messages:
            role = "model" if msg["role"] == "assistant" else msg["role"]
            if role == "system":
                if not system_instruction:
                    system_instruction = msg["content"]
                continue
            contents.append({"role": role, "parts": [{"text": msg["content"]}]})

        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
            }
        }
        if max_tokens:
            payload["generationConfig"]["maxOutputTokens"] = max_tokens
        if system_instruction:
            payload["systemInstruction"] = {"parts": [{"text": system_instruction}]}
            
        if extra_payload:
            payload.update(extra_payload)

        if stream:
            url += "?alt=sse"
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
                        candidates = chunk.get("candidates", [])
                        if candidates:
                            parts = candidates[0].get("content", {}).get("parts", [])
                            if parts:
                                delta = parts[0].get("text", "")
                                if delta:
                                    yield delta
                    except json.JSONDecodeError:
                        continue
