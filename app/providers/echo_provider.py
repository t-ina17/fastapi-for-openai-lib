"""Echo Provider

OpenAI互換のチャット出力を模した擬似プロバイダです。入力メッセージ（最後のuser）を
そのまま返すことで、APIフローやストリーミング挙動を素早く検証できます。

本プロバイダは推論処理を行わないため、CPU/GPUやモデルの準備が不要です。
"""

import asyncio
from typing import AsyncIterator, Dict, Any, Optional
from .base import GenerationProvider


class EchoProvider(GenerationProvider):
    """エコー応答を生成するプロバイダ。

    入力の`messages`から最後の`user`メッセージを抽出し、`Echo from {model}: ...`の
    形式で返します。`stream=True`の場合は小さなチャンクに分割してSSE用途の
    ストリーミング挙動を模倣します。
    """

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
        """テキスト生成の非同期イテレーターを返します。

        引数はOpenAI互換エンドポイントのパラメータに準拠しますが、
        本プロバイダでは`max_tokens`や`temperature`等は実際には使用しません。
        """
        last_user = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")
        reply = f"Echo from {model}: {last_user}"
        # simple chunking to simulate streaming
        chunk_size = 20
        for i in range(0, len(reply), chunk_size):
            await asyncio.sleep(0.01)
            yield reply[i : i + chunk_size]
