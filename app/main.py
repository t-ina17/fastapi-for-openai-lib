"""FastAPI application providing OpenAI互換の `/v1` エンドポイント。

主な機能:
- `GET /v1/models`: 利用可能モデル一覧（簡易）
- `POST /v1/chat/completions`: ChatCompletionレスポンス（非ストリーム/ストリームSSE）

認証は環境変数`APP_API_KEY`が設定されている場合にBearerトークンを要求します。
バックエンドは`APP_BACKEND`で切替可能（`echo`/`transformers`）。
"""

from __future__ import annotations
import time
import uuid
from typing import AsyncIterator, Optional

import orjson
from fastapi import FastAPI, Depends, Header, HTTPException, status
from fastapi.responses import JSONResponse, StreamingResponse

from .config import settings
from .schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionChoice,
    ChatMessage,
    ChatCompletionUsage,
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    DeltaMessage,
    ModelsList,
)
from .providers.echo_provider import EchoProvider

try:
    from .providers.transformers_provider import TransformersProvider  # optional
except Exception:  # pragma: no cover
    TransformersProvider = None  # type: ignore


app = FastAPI(title="OpenAI-Compatible Chat API for gpt-oss-20b")


def require_api_key(authorization: Optional[str] = Header(default=None)) -> None:
    """認証ヘッダを検証し、必要ならBearerトークンの一致を確認する。"""
    if settings.api_key:
        if not authorization or not authorization.lower().startswith("bearer "):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing Bearer token")
        token = authorization.split(" ", 1)[1].strip()
        if token != settings.api_key:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")


# Provider factory
if settings.backend == "echo":
    provider = EchoProvider()
elif settings.backend == "transformers":
    if TransformersProvider is None:
        raise RuntimeError("Transformers backend selected but not available.")
    provider = TransformersProvider(settings.model_id)
else:
    provider = EchoProvider()  # fallback to echo for unknown backends


@app.get("/health")
async def health():
    """簡易ヘルスチェック。"""
    return {"status": "ok"}


@app.get(f"{settings.api_prefix}/models", response_model=ModelsList, dependencies=[Depends(require_api_key)])
async def list_models():
    """OpenAI互換のモデル一覧（簡易）を返す。"""
    return ModelsList(data=[{"id": settings.model_id, "object": "model"}])


@app.post(f"{settings.api_prefix}/chat/completions", dependencies=[Depends(require_api_key)])
async def chat_completions(body: ChatCompletionRequest):
    """OpenAI互換のChat Completions。

    `stream: true`の場合はSSEでチャンクを逐次返却し、最後に`[DONE]`を送る。
    非ストリーム時は全文を生成して通常のJSONレスポンスを返す。
    """
    created = int(time.time())
    req = body

    max_tokens = req.max_tokens or settings.max_tokens
    temperature = req.temperature if req.temperature is not None else settings.temperature
    top_p = req.top_p if req.top_p is not None else settings.top_p
    stream = bool(req.stream)

    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

    async def generate_text() -> AsyncIterator[str]:
        """プロバイダからテキストトークンを非同期に取得するヘルパー。"""
        async for token in provider.generate(
            model=req.model,
            messages=[m.model_dump() for m in req.messages],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=stream,
            stop=req.stop,
        ):
            yield token

    if stream:
        async def event_stream() -> AsyncIterator[bytes]:
            # stream chunks as OpenAI SSE
            async for piece in generate_text():
                chunk = ChatCompletionChunk(
                    id=completion_id,
                    created=created,
                    model=req.model,
                    choices=[ChatCompletionChunkChoice(index=0, delta=DeltaMessage(content=piece))],
                )
                data = orjson.dumps(chunk.model_dump())
                yield b"data: " + data + b"\n\n"
            end = b"data: [DONE]\n\n"
            yield end

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    # Non-streaming: collect all
    full_text = ""
    async for piece in generate_text():
        full_text += piece

    choice = ChatCompletionChoice(
        index=0,
        message=ChatMessage(role="assistant", content=full_text),
        finish_reason="stop",
    )
    usage = ChatCompletionUsage(prompt_tokens=_approx_tokens(req), completion_tokens=_approx_count(full_text), total_tokens=0)
    usage.total_tokens = usage.prompt_tokens + usage.completion_tokens

    resp = ChatCompletionResponse(
        id=completion_id,
        created=created,
        model=req.model,
        choices=[choice],
        usage=usage,
    )
    return JSONResponse(resp.model_dump())


def _approx_tokens(req: ChatCompletionRequest) -> int:
    """トークン数の簡易近似。"""
    return sum(_approx_count(m.content) for m in req.messages)


def _approx_count(text: str) -> int:
    """~4文字/トークンの簡易近似。"""
    return max(1, int(len(text) / 4))
