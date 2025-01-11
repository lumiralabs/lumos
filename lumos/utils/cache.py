from typing import Callable, Any
from functools import wraps
import structlog
import sqlite3
import json
import asyncio
import hashlib

logger = structlog.get_logger()


def serialize_for_cache(obj: Any) -> str:
    """Helper function to serialize objects for caching"""
    if isinstance(obj, type):  # Handle classes/types first
        return obj.__name__
    elif hasattr(obj, "model_dump_json"):  # Then handle Pydantic model instances
        return obj.model_dump_json()
    elif isinstance(obj, (list, tuple)):  # Handle lists and tuples
        return json.dumps([serialize_for_cache(item) for item in obj])
    elif isinstance(obj, dict):  # Handle dictionaries
        return json.dumps({k: serialize_for_cache(v) for k, v in obj.items()})
    else:
        try:
            return json.dumps(obj)
        except TypeError:
            return str(obj)


def create_cache_key(func_name: str, args: tuple, kwargs: dict) -> str:
    """Create a stable cache key from function arguments"""
    # Serialize all arguments
    args_str = serialize_for_cache(args)
    kwargs_str = serialize_for_cache(kwargs)

    # Create a combined string
    combined = f"{func_name}:{args_str}:{kwargs_str}"

    # Create a hash of the combined string
    return hashlib.md5(combined.encode()).hexdigest()


def deserialize_from_cache(value: str, response_format: type | None = None) -> Any:
    """Helper function to deserialize cached values"""
    try:
        if response_format and hasattr(response_format, "model_validate_json"):
            return response_format.model_validate_json(value)
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return value


class LumosCache:
    def __init__(self, cache_name: str | None = None):
        if cache_name is None:
            cache_name = "lumos_cache"

        self.path = f"{cache_name}.db"

        self.conn = sqlite3.connect(self.path)
        self.cursor = self.conn.cursor()
        self.cursor.execute(
            "CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value TEXT)"
        )
        self.conn.commit()

    def get(self, key: str) -> str | None:
        self.cursor.execute("SELECT value FROM cache WHERE key = ?", (key,))
        result = self.cursor.fetchone()
        return result[0] if result else None

    def set(self, key: str, value: str):
        self.cursor.execute(
            "INSERT OR REPLACE INTO cache (key, value) VALUES (?, ?)", (key, value)
        )
        self.conn.commit()

    def __call__(self, func: Callable) -> Callable:
        """Makes the cache instance usable as a decorator"""

        if asyncio.iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                key = create_cache_key(func.__name__, args, kwargs)

                cached_result = self.get(key)
                if cached_result is not None:
                    logger.info("Cache hit", key=key)
                    response_format = kwargs.get("response_format")
                    return deserialize_from_cache(cached_result, response_format)

                result = await func(*args, **kwargs)

                try:
                    self.set(key, serialize_for_cache(result))
                except (sqlite3.Error, TypeError) as e:
                    logger.warning("Failed to cache result", error=str(e))

                return result

            return async_wrapper
        else:

            @wraps(func)
            def wrapper(*args, **kwargs):
                key = create_cache_key(func.__name__, args, kwargs)

                cached_result = self.get(key)
                if cached_result is not None:
                    logger.info("Cache hit", key=key)
                    response_format = kwargs.get("response_format")
                    return deserialize_from_cache(cached_result, response_format)

                result = func(*args, **kwargs)

                try:
                    self.set(key, serialize_for_cache(result))
                except (sqlite3.Error, TypeError) as e:
                    logger.warning("Failed to cache result", error=str(e))

                return result

            return wrapper
