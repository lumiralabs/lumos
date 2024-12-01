# lumos
Common utils across lumira labs

## Install
```
uv pip install git+https://github.com/lumiralabs/lumos
```


## Usage
```python
from lumos import lumos
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
```


## Deploy
We also expose lumos as a server
```
uv run uvicorn lumos.app:app
```
then curl:
```
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