# lumos
Simple utils for building AI apps. Available as python API and a server.


## Install
```bash
pip install git+https://github.com/lumiralabs/lumos
```


## Python API

### 1. Structured Outputs
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
# Response(steps=['Multiply 100 by 100.', '100 * 100 = 10000.'], final_answer='10000')
```

### 2. Embeddings API
```python
lumos.get_embedding("hello world")
#[0.12, ..., ..., 0.34]
```

### 3. Book Parser
```bash
python -m lumos.book.parser path/to/book.pdf
```
```
```python
from lumos import book_parser

book_parser.toc("path/to/book.pdf", level=2)
```
```
Table of Contents
├── Chapter 1. Introducing Asyncio (Pages: 1-8)
│   ├── The Restaurant of ThreadBots (Pages: 1-5)
│   ├── Epilogue (Pages: 6-5)
│   └── What Problem Is Asyncio Trying to Solve? (Pages: 6-8)
├── Chapter 2. The Truth About Threads (Pages: 9-20)
│   ├── Benefits of Threading (Pages: 10-10)
│   ├── Drawbacks of Threading (Pages: 11-13)
│   └── Case Study: Robots and Cutlery (Pages: 14-20)
├── Chapter 3. Asyncio Walk-Through (Pages: 21-74)
│   ├── Quickstart (Pages: 22-27)
│   ├── The Tower of Asyncio (Pages: 28-30)
│   ├── Coroutines (Pages: 31-36)
│   ├── Event Loop (Pages: 37-38)
│   ├── Tasks and Futures (Pages: 39-45)
│   ├── Async Context Managers: async with (Pages: 46-49)
│   ├── Async Iterators: async for (Pages: 50-52)
│   ├── Simpler Code with Async Generators (Pages: 53-54)
│   ├── Async Comprehensions (Pages: 55-56)
│   └── Starting Up and Shutting Down (Gracefully!) (Pages: 57-74)
├── Chapter 4. 20 Asyncio Libraries You Aren’t Using (But…Oh, Never Mind) (Pages: 75-128)
│   ├── Streams (Standard Library) (Pages: 76-87)
│   ├── Twisted (Pages: 88-90)
│   ├── The Janus Queue (Pages: 91-91)
│   ├── aiohttp (Pages: 92-97)
│   ├── ØMQ (ZeroMQ) (Pages: 98-109)
│   ├── asyncpg and Sanic (Pages: 110-125)
│   └── Other Libraries and Resources (Pages: 126-128)
└── Chapter 5. Concluding Thoughts (Pages: 129-130)
```
