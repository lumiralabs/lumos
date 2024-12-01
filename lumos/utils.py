import json
from typing import Callable
from functools import wraps

# Implementing middleware using decorators
def cache_middleware(func: Callable) -> Callable:
    """
    Cache middleware for Lumos functions.
    """
    cache = {}
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Serialize arguments to create a cache key
        messages = kwargs.get('messages', args[0] if len(args) > 0 else None)
        response_format = kwargs.get('response_format', args[1] if len(args) > 1 else None)
        examples = kwargs.get('examples', None)
        model = kwargs.get('model', "gpt-4o-mini")
        
        # Serialize messages
        messages_key = json.dumps(messages, sort_keys=True)
        
        # For response_format, we use its class name
        response_format_key = response_format.__name__
        
        # Serialize examples
        if examples:
            examples_key = json.dumps([(q, r.model_dump_json()) for q, r in examples], sort_keys=True)
        else:
            examples_key = None
        
        # Create a unique cache key
        key = (messages_key, response_format_key, examples_key, model)
        
        if key in cache:
            return cache[key]
        else:
            result = func(*args, **kwargs)
            cache[key] = result
            return result
        
    return wrapper
