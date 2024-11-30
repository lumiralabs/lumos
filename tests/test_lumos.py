def test_call_ai():
    from lumos import call_ai
    from pydantic import BaseModel
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
