from lumos import lumos

from pydantic import BaseModel


def test_call_ai():
    class Response(BaseModel):
        steps: list[str]
        final_answer: str

    resp = lumos.call_ai(
        messages=[
            {"role": "system", "content": "You are a mathematician."},
            {"role": "user", "content": "What is 100 * 100?"},
        ],
        response_format=Response,
        model="gpt-4o-mini",
    )

    assert resp.final_answer == "10000"


async def test_call_ai_async():
    class Response(BaseModel):
        steps: list[str]
        final_answer: str

    resp = await lumos.call_ai_async(
        messages=[
            {"role": "system", "content": "You are a mathematician."},
            {"role": "user", "content": "What is 100 * 100?"},
        ],
        response_format=Response,
        model="gpt-4o-mini",
    )
    assert resp.final_answer == "10000"
