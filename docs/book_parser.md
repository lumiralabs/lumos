---
title: Book Parser
---

The Book Parser extracts useful entities like:

1. Table of Contents
2. Section Data
3. Chunks


## Install
```bash
pip install lumira-lumos[book]
```

## 1. Table of Contents
```bash
python -m lumos.book.toc path/to/book.pdf
```
```python
from lumos import book_parser

book_parser.print_toc_from_pdf("path/to/book.pdf", level=2)
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


## 2. Section Data
```bash
python -m lumos.book.parser path/to/book.pdf sections
```
```python
from lumos import book_parser

sections = book_parser.section("path/to/book.pdf")
```

```
                                                                  Document Sections                                                                   
┏━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃    ┃        ┃                                                                   ┃                                                                  ┃
┃ ID ┃ Level  ┃ Title                                                             ┃ Content                                                          ┃
┃    ┃        ┃                                                                   ┃                                                                  ┃
┡━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│    │        │                                                                   │                                                                  │
│ 1  │ 1      │ Chapter 1. Introducing Asyncio                                    │ CHAPTER 1 Introducing Asyncio                                    │
│    │        │                                                                   │                                                                  │
│    │        │                                                                   │ My story is a lot like yours, only more interesting ’cause it    │
│    │        │                                                                   │ involves robots.                                                 │
│    │        │                                                                   │                                                                  │
│    │        │                                                                   │ —Bender, Futurama episode “30% Iron Chef ”                       │
│    │        │                                                                   │                                                                  │
│    │        │                                                                   │ The most common question I receive about Asyn...                 │
│    │        │                                                                   │                                                                  │
│    │        │                                                                   │                                                                  │
│ 2  │ 1.1    │ The Restaurant of ThreadBots                                      │ The Restaurant of ThreadBots The year is 2051, and you find      │
│    │        │                                                                   │ yourself in the restaurant business. Automation, largely by      │
│    │        │                                                                   │ robot workers, powers most of the economy, but it turns out that │
│    │        │                                                                   │ humans still en...                                               │
│    │        │                                                                   │                                                                  │
│    │        │                                                                   │                                                                  │
│ 3  │ 1.3    │ What Problem Is Asyncio Trying to Solve?                          │ What Problem Is Asyncio Trying to Solve? For I/O-bound           │
│    │        │                                                                   │ workloads, there are exactly (only!) two reasons to use          │
│    │        │                                                                   │ async-based concurrency over thread-based concurrency:           │
│    │        │                                                                   │                                                                  │
│    │        │                                                                   │ Asyncio offers a safer alternativ...                             │
│    │        │                                                                   │                                                                  │
│    │        │                                                                   │                                                                  │
│ 4  │ 2      │ Chapter 2. The Truth About Threads                                │ CHAPTER 2 The Truth About Threads                                │
│    │        │                                                                   │                                                                  │
│    │        │                                                                   │ Let’s be frank for a moment—you really don’t want to use Curio.  │
│    │        │                                                                   │ All things equal, you should probably be programming with        │
│    │        │                                                                   │ threads. Yes, threads. THOSE threads. Seri‐...                   │
│    │        │                                                                   │                                                                  │
│    │        │                                                                   │                                                                  │
│ 5  │ 2.1    │ Benefits of Threading                                             │ Benefits of Threading These are the main benefits of threading:  │
│    │        │                                                                   │                                                                  │
│    │        │                                                                   │ Ease of reading code                                             │
│    │        │                                                                   │                                                                  │
│    │        │                                                                   │ Your code can run concurrently, but still be set out in a very   │
│    │        │                                                                   │ simple, top-down linear sequence of commands to th...            │
```


## 3. Chunks
```bash
python -m lumos.book.parser path/to/book.pdf chunks
```
```python
from lumos import book_parser

chunks = book_parser.chunks("path/to/book.pdf")
```

```
                                                                   Document Chunks                                                                    
┏━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━┓
┃     ┃           ┃                                                                                                                           ┃      ┃
┃ #   ┃ Type      ┃ Text                                                                                                                      ┃ Page ┃
┃     ┃           ┃                                                                                                                           ┃      ┃
┡━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━┩
│     │           │                                                                                                                           │      │
│ 1   │ <no type> │ CHAPTER 1 Introducing Asyncio                                                                                             │ 13   │
│     │           │                                                                                                                           │      │
│     │           │ My story is a lot like yours, only more interesting ’cause it involves robots.                                            │      │
│     │           │                                                                                                                           │      │
│     │           │ —Bender, Futurama episode “30% Iron Chef ”                                                                                │      │
│     │           │                                                                                                                           │      │
│     │           │ The most common question I receive about Asyncio in Python 3 is this: “What is it, and what do I do with it?” The answer  │      │
│     │           │ you’ll hear most frequently is likely something about being able to execute multiple concurrent HTTP requests in a single │      │
│     │           │ program. But there is more to it than that—much more. Asyncio requires changing how you think about structuring programs. │      │
│     │           │                                                                                                                           │      │
│     │           │ The following story provides a backdrop for gaining this understanding. The central focus of Asyncio is on how best to    │      │
│     │           │ best perform multiple tasks at the same time—and not just any tasks, but specifically tasks that involve waiting periods. │      │
│     │           │ The key insight required with this style of programming is that while you wait for this task to com‐ plete, work on other │      │
│     │           │ tasks can be performed.                                                                                                   │      │
│     │           │                                                                                                                           │      │
│     │           │                                                                                                                           │      │
│ 2   │ <no type> │ Epilogue In our story, each robot worker in the restaurant is a single thread. The key observa‐ tion in the story is that │ 18   │
│     │           │ the nature of the work in the restaurant involves a great deal of waiting, just as requests.get() is waiting for a        │      │
│     │           │ response from a server.                                                                                                   │      │
│     │           │                                                                                                                           │      │
│     │           │ In a restaurant, the worker time spent waiting isn’t huge when slow humans are doing manual work, but when                │      │
│     │           │ super-efficient and quick robots are doing the work, nearly all their time is spent waiting. In computer programming, the │      │
│     │           │ same is true when net‐ work programming is involved. CPUs do work and wait on network I/O. CPUs in modern computers are   │      │
│     │           │ extremely fast—hundreds of thousands of times faster than network traffic. Thus, CPUs running networking programs spend a │      │
│     │           │ great deal of time waiting.                                                                                               │      │
│     │           │                                                                                                                           │      │
│     │           │                                                                                                                           │      │
│ 3   │ <no type> │ The insight in the story is that programs can be written to explicitly direct the CPU to move between work tasks as       │ 18   │
│     │           │ necessary. Although there is an improvement in econ‐ omy (using fewer CPUs for the same work), the real advantage,        │      │
│     │           │ compared to a threading (multi-CPU) approach, is the elimination of race conditions.                                      │      │
│     │           │                                                                                                                           │      │
│     │           │ It’s not all roses, however: as we found in the story, there are benefits and drawbacks to most technology solutions. The │      │
│     │           │ introduction of the LoopBot solved a certain class of problems but also introduced new problems—not the least of which is │      │
│     │           │ that the res‐ taurant owner had to learn a slightly different way of programming.                                         │      │
│     │           │                                                                                                                           │      │
│     │           │                                                                                                                           │      │
│ 4   │ <no type> │ The Restaurant of ThreadBots The year is 2051, and you find yourself in the restaurant business. Automation, largely by   │ 13   │
│     │           │ robot workers, powers most of the economy, but it turns out that humans still enjoy going out to eat once in a while. In  │      │
│     │           │ your restaurant, all the employees are robots—humanoid, of course, but unmistakably robots. The most successful manu‐     │      │
│     │           │ facturer of robots is Threading Inc., and robot workers from this company have come to be called “ThreadBots.”            │      │
│     │           │                                                                                                                           │      │
│     │           │ Except for this small robotic detail, your restaurant looks and operates like one of those old-time establishments from,  │      │
│     │           │ say, 2020. Your guests will be looking for that vintage experience. They want fresh food prepared from scratch. They want │      │
│     │           │ to sit at tables. They want to wait for their meals—but only a little. They want to pay at the end, and they sometimes    │      │
│     │           │ even want to leave a tip, for old times’ sake.                                                                            │      │
│     │           │                                                                                                                           │      │
│     │           │                                                                                                                           │      │
│ 5   │ <no type> │ Being new to the robotic restaurant business, you do what every other restaurateur does and hire a small fleet of robots: │ 14   │
│     │           │ one to greet diners at the front desk (GreetBot), one to wait tables and take orders (WaitBot), one to do the cooking     │      │
│     │           │ (ChefBot), and one to manage the bar (WineBot).                                                                           │      │
│     │           │                                                                                                                           │      │
│     │           │ Hungry diners arrive at the front desk and are welcomed by GreetBot, your front-of- house ThreadBot. They are then        │      │
│     │           │ directed to a table, and once they are seated, WaitBot takes their order. Then WaitBot brings that order to the kitchen   │      │
│     │           │ on a slip of paper (because you want to preserve that old-time experience, remember?). ChefBot looks at the order on the  │      │
│     │           │ slip and begins preparing the food. WaitBot will periodically check whether the food is ready, and when it is, will       │      │
│     │           │ immediately take the dishes to the cus‐ tomers’ table. When the guests are ready to leave, they return to GreetBot, who   │      │
│     │           │ calcu‐ lates the bill, takes their payment, and graciously wishes them a pleasant evening.                                │      │
```