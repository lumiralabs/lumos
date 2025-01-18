---
title: Server and Client
---

The entire Lumos API is also available as a server / client fashion to deploy and remotely call the API. 

A seperate service can be useful isolation for

1. Running compute heavy book parsing operations
2. A centralised AI server for your org

## Deploy (Server)
Simply host the FastAPI server, authenticated by an API key:
```bash
LUMOS_API_KEY=12345678 
uv run uvicorn lumos.server.app:app --host 0.0.0.0 --port 8000
```

## Client SDK
Once deployed, you can conveniently access the service with the `LumosClient` that fully mirrors the Python API.
```python
from lumos import lumos
```
can be safely replaced with 
```python
from lumos import LumosClient
lumos = LumosClient(host="http://localhost:8000", api_key="12345678")
```

Now you can do similar operations like:
```python
from pydantic import BaseModel

class Response(BaseModel):
    steps: list[str]
    final_answer: str


lumos.call_ai(
    messages=[
        {"role": "system", "content": "You are a mathematician."},
        {"role": "user", "content": "What is 100 * 100?"},
    ],
    response_format=Response,
    model="gpt-4o-mini",
)
# Response(steps=['Multiply 100 by 100.', '100 * 100 = 10000.'], final_answer='10000')
```
