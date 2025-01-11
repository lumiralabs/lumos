import asyncio
import hashlib
import json
import sqlite3
from functools import wraps
from typing import Callable, Any


def serialize_for_cache(obj: Any) -> str:
    """
    Serialize objects to a string for caching. This handles:
      - Pydantic models (via `model_dump_json()`)
      - lists/tuples of floats/ints (embedded vectors)
      - arbitrary JSON-serializable objects
      - fallback to str(obj)
    """
    if isinstance(obj, type):  # classes/types
        return obj.__name__
    elif hasattr(obj, "model_dump_json"):  # Pydantic
        return obj.model_dump_json()
    elif isinstance(obj, (list, tuple)):
        # Handle embeddings or nested embeddings
        if all(isinstance(x, (float, int)) for x in obj):
            return json.dumps({"__type__": "embedding", "data": list(obj)})
        elif all(
            isinstance(x, list) and all(isinstance(y, (float, int)) for y in x)
            for x in obj
        ):
            return json.dumps(
                {"__type__": "embeddings", "data": [list(x) for x in obj]}
            )
        return json.dumps([serialize_for_cache(item) for item in obj])
    elif isinstance(obj, dict):
        return json.dumps({k: serialize_for_cache(v) for k, v in obj.items()})
    else:
        try:
            return json.dumps(obj)
        except TypeError:
            return str(obj)


def create_cache_key(func_name: str, args: tuple, kwargs: dict) -> str:
    """
    Create a stable cache key from function arguments.
    """
    args_str = serialize_for_cache(args)
    kwargs_str = serialize_for_cache(kwargs)
    combined = f"{func_name}:{args_str}:{kwargs_str}"
    return hashlib.md5(combined.encode()).hexdigest()


def deserialize_from_cache(value: str, response_format: type | None = None) -> Any:
    """
    Deserialize cached values back to Python objects (including Pydantic if needed).
    """
    try:
        if response_format and hasattr(response_format, "model_validate_json"):
            # For Pydantic v2
            return response_format.model_validate_json(value)

        parsed = json.loads(value)

        # Handle embedding special-case
        if isinstance(parsed, dict) and "__type__" in parsed:
            if parsed["__type__"] == "embedding":
                return parsed["data"]
            elif parsed["__type__"] == "embeddings":
                return parsed["data"]
        return parsed

    except (json.JSONDecodeError, TypeError):
        return value


class LumosCache:
    """
    Persistent SQLite cache.
    """

    def __init__(self, cache_name: str = ""):
        self.path = f".lumoscache_{cache_name}.db"
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
        """
        Decorator that caches sync or async functions depending on their type.
        """

        if asyncio.iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                key = create_cache_key(func.__name__, args, kwargs)
                cached_result = self.get(key)
                if cached_result is not None:
                    return deserialize_from_cache(
                        cached_result, kwargs.get("response_format")
                    )

                # If not cached, compute
                result = await func(*args, **kwargs)
                try:
                    self.set(key, serialize_for_cache(result))
                except (sqlite3.Error, TypeError):
                    pass  # optionally log a warning

                return result

            return async_wrapper

        else:

            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                key = create_cache_key(func.__name__, args, kwargs)
                cached_result = self.get(key)
                if cached_result is not None:
                    return deserialize_from_cache(
                        cached_result, kwargs.get("response_format")
                    )

                # If not cached, compute
                result = func(*args, **kwargs)
                try:
                    self.set(key, serialize_for_cache(result))
                except (sqlite3.Error, TypeError):
                    pass  # optionally log a warning

                return result

            return sync_wrapper
