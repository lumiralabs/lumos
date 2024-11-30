# lumos
Common utils across lumira labs


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