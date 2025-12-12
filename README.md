# fastapi-for-openai-lib

OpenAI互換の `/v1` エンドポイントを提供する FastAPI サーバーです。モデルID `gpt-oss-20b` を返し、OpenAI公式ライブラリ(>=1.x)から利用できます。

このリポジトリは3つのバックエンドモードを想定しています。
- `echo` (デフォルト): 実際の生成はしませんがOpenAI互換の形で返します。
- `transformers`: Hugging Face Transformers でローカル推論します（大規模モデルはGPU/メモリ必須）。
- `vllm_proxy` (将来拡張): vLLMのOpenAI互換サーバーへプロキシします。

## クイックスタート（uv + echo バックエンド）

1) uv のインストール（未導入の場合）

```bash
# Homebrew
brew install uv
# もしくは公式インストーラ
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2) 仮想環境の作成と依存関係のインストール（uv）

```bash
cd fastapi-for-openai-lib
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

3) 環境変数（任意でAPIキーを要求）

```bash
# 任意: ベアラートークンを要求したい場合
export APP_API_KEY="sk-local-123"
# 任意: モデルID（表示用）
export APP_MODEL_ID="gpt-oss-20b"
# 任意: バックエンド
export APP_BACKEND="echo"  # echo | transformers
```

4) サーバー起動（uv 経由）

```bash
uv run uvicorn app.main:app --reload --port 8000
```

5) 動作確認（curl）

```bash
curl -s \
  -H "Authorization: Bearer sk-local-123" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-oss-20b",
    "messages": [{"role":"user","content":"こんにちは！"}]
  }' \
  http://localhost:8000/v1/chat/completions | jq
```

### OpenAI Python ライブラリからの利用

OpenAI SDK v1 以降:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="sk-local-123",  # APP_API_KEY と合わせる
)

resp = client.chat.completions.create(
    model="gpt-oss-20b",
    messages=[{"role": "user", "content": "こんにちは！"}],
)
print(resp.choices[0].message.content)
```

ストリーミング:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="sk-local-123")

with client.chat.completions.stream(
    model="gpt-oss-20b",
    messages=[{"role":"user","content":"自己紹介して"}],
) as stream:
    for event in stream:
        if event.type == "completion.delta":
            print(event.delta, end="", flush=True)
```

OpenAI クライアントを使う場合は別途インストール:

```bash
uv pip install openai
```

## Transformers バックエンド

GPUがなくてもCPUで動作可能ですが、推論速度はかなり遅くなります。なるべく小さめのモデルIDを選択してください（例: 7B未満）。macOSではMPS(GPU)があれば利用され、無ければ自動でCPUにフォールバックします。

```bash
uv pip install "transformers>=4.44" "accelerate>=1.0"
# torchは環境に合わせて https://pytorch.org/get-started/locally/ を参照（CPUのみでも可）
# 例: CPU版torch (macOS/Linux)
uv pip install torch --index-url https://download.pytorch.org/whl/cpu

export APP_BACKEND=transformers
# 実在する小さめのHFモデルIDに置換（CPUならまずは gpt2 など）
export APP_MODEL_ID=gpt2
uv run uvicorn app.main:app --port 8000
```

注意: `openai/gpt-oss-20b` はプレースホルダー名です。実際のHugging FaceモデルID（例: Llama系、Mistral系など）を指定してください。GPU/MPSがない場合はCPU実行になります（本実装は未検出時に`device_map="cpu"`、float32でロード）。

### 動作確認（OpenAI互換; CPUで gpt2 の例）
```bash
curl -s \
  -H "Authorization: Bearer $APP_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt2",
    "messages": [{"role":"user","content":"こんにちは！"}]
  }' \
  http://localhost:8000/v1/chat/completions | jq
```

### まずはCPUで安全に試す方法
- `APP_BACKEND=echo` でOpenAI互換形のみ確認（高速）
- CPUで `APP_BACKEND=transformers` に切替。小さいモデルでレスポンス形を確認
- 実運用は vLLM などの専用サーバーを検討すると高速化しやすいです

## エンドポイント

- `GET /v1/models`: `[{ id: APP_MODEL_ID }]` を返却
- `POST /v1/chat/completions`: OpenAI互換のChatCompletionレスポンス
  - `stream: true` の場合はSSEでチャンクを返却し、最後に`[DONE]`

## 認証

`APP_API_KEY` を設定した場合、`Authorization: Bearer <key>` が必須になります。

## ライセンス

このプロジェクトはサンプル実装です。各モデルの使用条件・ライセンス、推論環境の利用規約に従ってください。
