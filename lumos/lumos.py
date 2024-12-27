from litellm import completion, acompletion, embedding, transcription
import json
from typing import Any, TypeVar
from pydantic import BaseModel
from lumos.utils import cache_middleware
import base64
import structlog
import magic

logger = structlog.get_logger()

T = TypeVar('T', bound=BaseModel)


def _construct_chat_examples(examples: list[tuple[str, T]], schema: type[T]) -> list[dict[str, str]]:
    '''
    Construct a list of chat messages from a list of examples.
    Examples are pairs of query: response. Response should be the pydantic model 
    '''
    chat_messages: list[dict[str, str]] = []
    for query, response in examples:
        chat_messages.append({"role": "user", "content": query})
        chat_messages.append({"role": "assistant", "content": response.model_dump_json()})
    return chat_messages


@cache_middleware
def call_ai(messages: list[dict[str, str]], response_format: type[T] | None = None, examples: list[tuple[str, T]] | None = None, model="gpt-4o-mini"):
    '''
    Make an AI completion call using litellm, with support for few-shot examples.

    Args:
        messages: list of chat messages to send. If examples are provided, must contain exactly 2 messages - a system message and a user message.
        response_format: A Pydantic model class defining the expected response structure.
        examples: Optional list of (query, response) tuples for few-shot learning. Each response
                 must be an instance of the response_format class (converted to json)
        model: defaults to "gpt-4o-mini"

    Returns:
        An instance of the response_format class containing the AI's response.

    Example:
        from pydantic import BaseModel

        class MathResponse(BaseModel):
            answer: int
            explanation: str

        # Basic usage
        messages = [
            {"role": "system", "content": "You are a math helper"},
            {"role": "user", "content": "What is 2+2?"}
        ]
        response = call_ai(messages, MathResponse)

        # With examples
        add_examples = [
            ("What is 3+3?", MathResponse(answer=6, explanation="3 plus 3 equals 6")),
            ("What is 4+4?", MathResponse(answer=8, explanation="4 plus 4 equals 8"))
        ]
        response = call_ai(messages, MathResponse, examples=add_examples)
        # MathResponse(answer=6, explanation="3 plus 3 equals 6")
    '''
    # Prepare messages with examples if provided
    if examples:
        example_messages = _construct_chat_examples(examples, response_format)
        assert len(messages) <= 2, "Can only have up to 2 messages when using examples"
        assert messages[0]["role"] == "system", "First message must be system message when using examples"
        assert messages[1]["role"] == "user", "Second message must be user message when using examples"
        _messages = [messages[0]] + example_messages + [messages[1]]
    else:
        _messages = messages

    # Make the AI completion call
    response = completion(
        model=model,
        messages=_messages,
        response_format=response_format
    )
    cost = response._hidden_params['response_cost']
    logger.info("ai_cost", cost=cost)
    ret = response.choices[0]['message']['content']
    if response_format:
        ret_dict = json.loads(ret)
        ret_obj = response_format.model_validate(ret_dict)
        return ret_obj
    
    return ret

@cache_middleware
async def call_ai_async(messages: list[dict[str, str]], response_format: type[T] | None = None, examples: list[tuple[str, T]] | None = None, model="gpt-4o-mini"):
    '''
    Make an AI completion call using litellm, with support for few-shot examples.

    Args:
        messages: list of chat messages to send. If examples are provided, must contain exactly 2 messages - a system message and a user message.
        response_format: A Pydantic model class defining the expected response structure.
        examples: Optional list of (query, response) tuples for few-shot learning. Each response
                 must be an instance of the response_format class (converted to json)
        model: defaults to "gpt-4o-mini"

    Returns:
        An instance of the response_format class containing the AI's response.

    Example:
        from pydantic import BaseModel

        class MathResponse(BaseModel):
            answer: int
            explanation: str

        # Basic usage
        messages = [
            {"role": "system", "content": "You are a math helper"},
            {"role": "user", "content": "What is 2+2?"}
        ]
        response = await call_ai_async(messages, MathResponse)

        # With examples
        add_examples = [
            ("What is 3+3?", MathResponse(answer=6, explanation="3 plus 3 equals 6")),
            ("What is 4+4?", MathResponse(answer=8, explanation="4 plus 4 equals 8"))
        ]
        response = await call_ai_async(messages, MathResponse, examples=add_examples)
        # MathResponse(answer=6, explanation="3 plus 3 equals 6")
    '''
    # Prepare messages with examples if provided
    if examples:
        example_messages = _construct_chat_examples(examples, response_format)
        assert len(messages) <= 2, "Can only have up to 2 messages when using examples"
        assert messages[0]["role"] == "system", "First message must be system message when using examples"
        assert messages[1]["role"] == "user", "Second message must be user message when using examples"
        _messages = [messages[0]] + example_messages + [messages[1]]
    else:
        _messages = messages

    # Make the AI completion call
    response = await acompletion(
        model=model,
        messages=_messages,
        response_format=response_format
    )   
    cost = response._hidden_params['response_cost']
    logger.info("ai_cost", cost=cost)
    ret = response.choices[0]['message']['content']
    if response_format:
        ret_dict = json.loads(ret)
        ret_obj = response_format.model_validate(ret_dict)
        return ret_obj
    
    return ret



def get_embedding(text: str | list[str], model: str = "text-embedding-3-small"):
    if isinstance(text, str):
        _text = [text]
    else:
        _text = text
    
    embeddings = embedding(model, _text).json()['data']
    if isinstance(text, str):
        return embeddings[0]['embedding']
    else:
        return [e['embedding'] for e in embeddings]

def transcribe(file, model: str = "whisper-1"):
    return transcription(file, model)

async def describe_image(image: bytes, model: str = "gpt-4o-mini") -> str | None:
    # Validate image
    if not isinstance(image, bytes):
        logger.warning("Image must be bytes")
        return None
        
    # Validate size
    if len(image) > 20 * 1024 * 1024:  # 20MB limit
        logger.warning("Image too large") 
        return None

    # Detect mimetype
    mime = magic.Magic(mime=True)
    detected_mime = mime.from_buffer(image)
    
    if detected_mime not in ['image/png', 'image/jpeg', 'image/webp']:
        logger.warning(f"Unsupported image mimetype: {detected_mime}")
        return None

    # Encode as base64
    img_base64 = base64.b64encode(image).decode("utf-8")

    return await call_ai_async(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image in detail:"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{detected_mime};base64,{img_base64}"
                        },
                    },
                ],
            }
        ],
        model=model
    )

def get_knn(query: str, vec_db: Any, k: int = 10):
    '''
    Get k-nearest neighbors chunks for a query and a given vector store
    '''
    # TODO (jay): implement
    pass
