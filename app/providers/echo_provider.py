import asyncio
from typing import AsyncIterator, Dict, Any, Optional
from .base import GenerationProvider


class EchoProvider(GenerationProvider):
    async def generate(
        self,
        *,
        model: str,
        messages: list[Dict[str, str]],
        max_tokens: int,
        temperature: float,
        top_p: float,
        stream: bool,
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        last_user = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")
        reply = f"Echo from {model}: {last_user}"
        # simple chunking to simulate streaming
        chunk_size = 20
        for i in range(0, len(reply), chunk_size):
            await asyncio.sleep(0.01)
            yield reply[i : i + chunk_size]
