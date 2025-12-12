from openai import OpenAI
import os

base_url = os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1")
api_key = os.environ.get("OPENAI_API_KEY", "sk-local-123")

client = OpenAI(base_url=base_url, api_key=api_key)

resp = client.chat.completions.create(
    model=os.environ.get("OPENAI_MODEL", "gpt-oss-20b"),
    messages=[{"role": "user", "content": "こんにちは！"}],
)
print(resp.choices[0].message.content)
