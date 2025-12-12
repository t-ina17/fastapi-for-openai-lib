"""OpenAI互換のスキーマ定義。

ChatCompletion APIのリクエスト/レスポンス、ストリームチャンク、モデル一覧
などをPydanticモデルとして定義します。
"""

from __future__ import annotations
from typing import List, Optional, Literal, Any
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str


class ChatCompletionRequest(BaseModel):
    """OpenAI互換のチャット補完リクエスト。"""
    model: str
    messages: List[ChatMessage]
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    stream: Optional[bool] = False
    stop: Optional[List[str]] = None
    # passthrough unsupported fields
    n: Optional[int] = 1
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None


class ChatCompletionChoice(BaseModel):
    """非ストリーム時の選択肢。"""
    index: int
    message: ChatMessage
    finish_reason: Optional[str] = None


class ChatCompletionUsage(BaseModel):
    """トークン使用量の簡易情報。"""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    """OpenAI互換のチャット補完レスポンス。"""
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Optional[ChatCompletionUsage] = None


class DeltaMessage(BaseModel):
    """ストリームチャンクで用いられる差分メッセージ。"""
    role: Optional[str] = None
    content: Optional[str] = None


class ChatCompletionChunkChoice(BaseModel):
    """ストリームチャンクの選択肢。"""
    index: int
    delta: DeltaMessage
    finish_reason: Optional[str] = None


class ChatCompletionChunk(BaseModel):
    """OpenAI互換のストリームチャンク(SSE)。"""
    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str
    choices: List[ChatCompletionChunkChoice]


class ModelsList(BaseModel):
    """モデル一覧レスポンス。"""
    object: Literal["list"] = "list"
    data: list[dict[str, Any]]
