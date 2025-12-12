from typing import AsyncIterator, Dict, Any, Optional


class GenerationProvider:
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
        raise NotImplementedError
