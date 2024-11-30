from litellm import completion
import json

def call_ai(model, messages, response_format=None):
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


