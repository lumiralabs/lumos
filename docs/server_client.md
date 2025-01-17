---
title: Server and Client
---

## Deploy 
We also expose lumos as a server authenticated by an API key.
```
uv run uvicorn lumos.server.app:app
```
then curl:
```bash
curl -X POST "http://localhost:8000/gen" \
-H "Content-Type: application/json" \
-H "X-API-Key: 12345678" \
-d '{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant"},
    {"role": "user", "content": "What is the capital of France?"}
  ],
  "response_schema": {
    "type": "object",
    "properties": {
      "final_answer": {
        "title": "Final Answer",
        "type": "string"
      }
    },
    "required": ["final_answer"],
    "title": "Response"
  },
  "model": "gpt-4o-mini"
}'
```


Use the python client to access lumos APIs remotely

```python
from lumos import LumosClient

lumos = LumosClient("http://localhost:8000", "12345678")

await lumos.call_ai_async(
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "What is the capital of France?"}
    ],
    response_schema={
        "type": "object",
        "properties": {
            "final_answer": {
                "title": "Final Answer",
                "type": "string"
            }
        },
        "required": ["final_answer"],
        "title": "Response"
    },
    model="gpt-4o-mini"
)
```
