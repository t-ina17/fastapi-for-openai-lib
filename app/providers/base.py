"""生成プロバイダのベースインターフェース。

各バックエンドはこの抽象インターフェースに従い、与えられたチャット
メッセージから非同期にテキストトークンを生成・ストリーミングします。
"""

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
        """非同期トークン生成のジェネレーターを返す抽象メソッド。"""
        raise NotImplementedError
