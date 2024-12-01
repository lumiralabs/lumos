from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, create_model
from typing import Literal, Any, Callable
from lumos.lumos import call_ai
from functools import wraps

app = FastAPI(title="Lumos API")

def require_api_key(func: Callable):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        request: Request = kwargs.get('request') or args[0]
        api_key = request.headers.get('X-API-Key')
        if not api_key:
            raise HTTPException(status_code=401, detail="API key is missing")
        if not api_key.strip():
            raise HTTPException(status_code=401, detail="Invalid API key")
        if api_key != "12345678":
            raise HTTPException(status_code=401, detail="Invalid API key")
        return await func(*args, **kwargs)
    return wrapper

class ChatMessage(BaseModel):
    role: str = Literal["system", "user", "assistant"]
    content: str

class AIRequest(BaseModel):
    messages: list[ChatMessage]
    response_schema: dict[str, Any]  # JSON schema format
    examples: list[tuple[str, dict[str, Any]]] | None = None
    model: str | None = "gpt-4o-mini"

def _json_schema_to_pydantic_types(schema: dict[str, Any]) -> dict[str, tuple[type, Any]]:
    """Convert JSON schema types to Python/Pydantic types"""
    type_mapping = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "array": list,
        "object": dict
    }
    
    field_types = {}
    properties = schema.get("properties", {})
    required = schema.get("required", [])
    
    for field_name, field_schema in properties.items():
        python_type = type_mapping[field_schema["type"]]
        # If field is required, use ... as default, otherwise None
        default = ... if field_name in required else None
        field_types[field_name] = (python_type, default)
    
    return field_types

@app.post("/gen")
@require_api_key
async def create_chat_completion(request: Request, ai_request: AIRequest):
    try:
        # Convert JSON schema to Pydantic field types
        field_types = _json_schema_to_pydantic_types(ai_request.response_schema)
        
        # Dynamically create a Pydantic model
        ResponseModel = create_model(
            'DynamicResponseModel',
            **field_types
        )
        
        # Convert examples if provided
        formatted_examples = None
        if ai_request.examples:
            formatted_examples = [
                (query, ResponseModel(**response))
                for query, response in ai_request.examples
            ]
        
        # Convert messages to dict format
        messages = [msg.model_dump() for msg in ai_request.messages]
        
        # Call the AI function
        result = call_ai(
            messages=messages,
            response_format=ResponseModel,
            examples=formatted_examples,
            model=ai_request.model
        )
        
        return result.model_dump()
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/healthz")
@require_api_key
async def health_check(request: Request):
    return {"status": "healthy"}

@app.get("/")
async def root(request: Request):
    return {"message": "Lumos API"}