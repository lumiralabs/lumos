import json
import hashlib
from typing import Callable
from functools import wraps


# Implementing middleware using decorators
def cache_middleware(func: Callable) -> Callable:
    """
    Cache middleware for Lumos functions.
    """
    cache = Cache()

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Serialize arguments to create a cache key
        messages = kwargs.get("messages", args[0] if len(args) > 0 else None)
        response_format = kwargs.get(
            "response_format", args[1] if len(args) > 1 else None
        )
        examples = kwargs.get("examples", None)
        model = kwargs.get("model", "gpt-4o-mini")

        # Create components for the cache key
        key_components = {
            "messages": messages,
            "response_format": response_format.__name__ if response_format else None,
            "examples": [(q, r.model_dump()) for q, r in examples]
            if examples
            else None,
            "model": model,
        }

        # Create a unique cache key
        key_str = json.dumps(key_components, sort_keys=True)
        cache_key = hashlib.sha256(key_str.encode()).hexdigest()

        # Check cache
        cached_result = cache.get(cache_key)
        if cached_result is not None:
            if response_format:
                return response_format.model_validate(cached_result)
            return cached_result

        # If not in cache, call function and cache result
        result = func(*args, **kwargs)

        # Store in cache
        cache_value = result.model_dump() if hasattr(result, "model_dump") else result
        cache.set(cache_key, cache_value)

        return result

    return wrapper


def async_cache_middleware(func: Callable) -> Callable:
    """
    Persistent cache middleware for async Lumos functions.
    """
    cache = Cache()

    @wraps(func)
    async def wrapper(*args, **kwargs):
        # Serialize arguments to create a cache key
        messages = kwargs.get("messages", args[0] if len(args) > 0 else None)
        response_format = kwargs.get(
            "response_format", args[1] if len(args) > 1 else None
        )
        examples = kwargs.get("examples", None)
        model = kwargs.get("model", "gpt-4o-mini")

        # Create components for the cache key
        key_components = {
            "messages": messages,
            "response_format": response_format.__name__ if response_format else None,
            "examples": [(q, r.model_dump()) for q, r in examples]
            if examples
            else None,
            "model": model,
        }

        # Create a unique cache key
        key_str = json.dumps(key_components, sort_keys=True)
        cache_key = hashlib.sha256(key_str.encode()).hexdigest()

        # Check cache
        cached_result = cache.get(cache_key)
        if cached_result is not None:
            if response_format:
                return response_format.model_validate(cached_result)
            return cached_result

        # If not in cache, call function and cache result
        result = await func(*args, **kwargs)

        # Store in cache
        cache_value = result.model_dump() if hasattr(result, "model_dump") else result
        cache.set(cache_key, cache_value)

        return result

    return wrapper


class Cache:
    def __init__(self):
        self.cache_file = ".lumos_cache.json"
        # Load existing cache or create new one
        try:
            with open(self.cache_file, "r") as f:
                self.cache = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            self.cache = {}

    def get(self, key: str) -> dict | None:
        return self.cache.get(key)

    def set(self, key: str, value: dict) -> None:
        self.cache[key] = value
        # Persist to disk
        with open(self.cache_file, "w") as f:
            json.dump(self.cache, f)
