from lumos.lumos import construct_chat_examples, call_ai
from pydantic import BaseModel

def test_construct_chat_examples():
    class Response(BaseModel):
        steps: list[str]
        final_answer: str

    chat_messages = construct_chat_examples([("What is 100 * 100?", Response(steps=["100 * 100 = 10000"], final_answer="10000"))], Response)
    assert chat_messages == [{"role": "user", "content": "What is 100 * 100?"}, {"role": "assistant", "content": '{"steps": ["100 * 100 = 10000"], "final_answer": "10000"}'}]


def test_call_ai():
    class Response(BaseModel):
        steps: list[str]
        final_answer: str

    resp = call_ai(
        "gpt-4o-mini", 
        [{"role": "system", "content": "You are a mathematician."}, 
         {"role": "user", "content": "What is 100 * 100?"}], 
        Response
    )
    
    assert resp.final_answer == "10000"
