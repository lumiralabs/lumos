from litellm import completion
import json
from typing import Any
def call_ai(messages, response_format, model="gpt-4o-mini"):
    '''
    Wrapper around litellm completion
    '''
    response = completion(
      model=model,
      messages=messages,
      response_format=response_format
    )
    resp_json = response.choices[0]['message']['content']
    resp_dict = json.loads(resp_json)
    return response_format.model_validate(resp_dict)



def get_knn(query: str, vec_db: Any, k: int = 10):
    '''
    Get k-nearest neighbors chunks for a query and a given vector store
    '''
    # TODO (jay): implement
    pass
