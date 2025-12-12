"""Transformers Provider

Hugging Face Transformers を用いてローカル推論を行うプロバイダ。
GPU/MPSがあれば自動利用し、未検出時はCPUでfloat32ロードにフォールバックします。
"""

from typing import AsyncIterator, Dict, Any, Optional
import asyncio

from .base import GenerationProvider


class TransformersProvider(GenerationProvider):
    def __init__(self, model_id: str):
        self.model_id = model_id
        self._pipeline = None
        self._tokenizer = None

    def _ensure_loaded(self):
        """モデル/トークナイザ/パイプラインの遅延ロードを行う。"""
        if self._pipeline is not None:
            return
        try:
            from transformers import AutoTokenizer, TextIteratorStreamer, AutoModelForCausalLM
            import torch
        except Exception as e:
            raise RuntimeError(
                "Transformers backend not available. Install transformers and torch (CPU or GPU)."
            ) from e

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id, use_fast=True)
        # CPU優先（GPU/MPSが無ければCPU）。CPUの場合はfloat32でロード。
        has_mps = False
        try:
            # macOS Metal Performance Shaders (MPS)
            has_mps = hasattr(torch, "backends") and hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        except Exception:
            has_mps = False

        use_gpu = torch.cuda.is_available() or has_mps
        if use_gpu:
            torch_dtype = getattr(torch, "bfloat16", None) or getattr(torch, "float16", None)
            model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch_dtype,
                device_map="auto",
            )
        else:
            # CPU: avoid accelerate offload; load normally on CPU with float32
            model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=getattr(torch, "float32", None) or None,
            )
        from transformers import pipeline

        # Build pipeline. If accelerate/device_map was used (GPU/MPS), omit `device`.
        if use_gpu:
            self._pipeline = pipeline(
                "text-generation",
                model=model,
                tokenizer=self._tokenizer,
            )
        else:
            # CPU pipeline explicitly uses device=-1
            self._pipeline = pipeline(
                "text-generation",
                model=model,
                tokenizer=self._tokenizer,
                device=-1,
            )

    def _build_prompt(self, messages: list[Dict[str, str]]) -> str:
        """Chatテンプレートがあれば利用し、なければ簡易連結でプロンプト生成。"""
        tok = self._tokenizer
        if tok is not None and hasattr(tok, "apply_chat_template"):
            try:
                return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                pass
        # Fallback simple concatenation
        parts = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            parts.append(f"{role}: {content}")
        parts.append("assistant:")
        return "\n".join(parts)

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
        self._ensure_loaded()

        prompt = self._build_prompt(messages)

        if stream:
            from transformers import TextIteratorStreamer
            import threading

            streamer = TextIteratorStreamer(self._tokenizer, skip_prompt=True, skip_special_tokens=True)
            gen_kwargs = dict(
                text_inputs=prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                streamer=streamer,
            )

            def _run():
                self._pipeline(**gen_kwargs)

            thread = threading.Thread(target=_run)
            thread.start()
            async for token in _aiter_streamer(streamer):
                yield token
            thread.join()
        else:
            # non-streaming: produce once, but still yield once for interface
            outputs = self._pipeline(
                prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
            )
            text = outputs[0]["generated_text"]
            # remove prompt prefix if present
            if text.startswith(prompt):
                text = text[len(prompt) :]
            yield text


async def _aiter_streamer(streamer):
    """BlockingなTextIteratorStreamerを非同期イテレーターへ橋渡しする。"""
    # Bridge blocking iterator to async iterator
    loop = asyncio.get_event_loop()
    while True:
        token = await loop.run_in_executor(None, _next_or_none, streamer)
        if token is None:
            break
        yield token


def _next_or_none(it):
    """`next(it)`のStopIterationをNoneで表現するヘルパー。"""
    try:
        return next(it)
    except StopIteration:
        return None
