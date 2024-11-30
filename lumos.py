from litellm import completion
import json

def call_ai(model, messages, response_format):
    '''
    Wrapper around litellm completion with additionl utils.
    '''
    response = completion(
      model=model,
      messages=messages,
      response_format=response_format
    )
    resp_json = response.choices[0]['message']['content']
    resp_dict = json.loads(resp_json)
    return response_format.model_validate(resp_dict)


def test_call_ai():
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
