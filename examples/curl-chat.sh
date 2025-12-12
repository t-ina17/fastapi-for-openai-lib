#!/usr/bin/env zsh

: ${OPENAI_API_KEY:=sk-local-123}
: ${OPENAI_BASE_URL:=http://localhost:8000/v1}

curl -s \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-oss-20b",
    "messages": [{"role":"user","content":"自己紹介して"}],
    "stream": false
  }' \
  "$OPENAI_BASE_URL/chat/completions" | jq
