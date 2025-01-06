import json
import hashlib
from typing import Callable
from functools import wraps
import os
import time
import percache
import structlog

logger = structlog.get_logger()


def init_cache(cache_path: str) -> percache.Cache:
    """Initialize a percache Cache."""
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    return percache.Cache(cache_path)


# Implementing middleware using decorators
def cache_middleware(func: Callable) -> Callable:
    """
    Cache middleware for Lumos functions using percache.
    """
    cache = init_cache(".lumos_cache")

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

        try:
            # Check cache
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                if response_format:
                    return response_format.model_validate(cached_result)
                return cached_result

            # If not in cache, call function and cache result
            result = func(*args, **kwargs)

            # Store in cache
            cache_value = (
                result.model_dump() if hasattr(result, "model_dump") else result
            )
            cache.set(cache_key, cache_value)

            return result
        except Exception as e:
            logger.error("Cache error", error=str(e), func=func.__name__)
            return func(*args, **kwargs)

    return wrapper


def async_cache_middleware(func: Callable) -> Callable:
    """
    Persistent cache middleware for async Lumos functions using percache.
    """
    cache = init_cache(".lumos_cache")

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

        try:
            # Check cache
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                if response_format:
                    return response_format.model_validate(cached_result)
                return cached_result

            # If not in cache, call function and cache result
            result = await func(*args, **kwargs)

            # Store in cache
            cache_value = (
                result.model_dump() if hasattr(result, "model_dump") else result
            )
            cache.set(cache_key, cache_value)

            return result
        except Exception as e:
            logger.error("Cache error", error=str(e), func=func.__name__)
            return await func(*args, **kwargs)

    return wrapper


class FileCache:
    def __init__(self, cache_dir: str = ".lumos_file_cache"):
        os.makedirs(cache_dir, exist_ok=True)
        self.cache = init_cache(os.path.join(cache_dir, "file_cache"))

    def _get_file_hash(self, file_path: str) -> str:
        """Calculate SHA-256 hash of a file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            # Read the file in chunks to handle large files
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def get(self, file_path: str) -> str | None:
        """
        Get cached content for a file if it exists.
        Returns the markdown content if cached, None otherwise.
        """
        file_hash = self._get_file_hash(file_path)
        cache_entry = self.cache.get(file_hash)
        if cache_entry is None:
            return None
        return cache_entry["content"]

    def set(self, file_path: str, content: str) -> None:
        """
        Cache the markdown content for a file.
        """
        file_hash = self._get_file_hash(file_path)
        self.cache.set(file_hash, {"content": content, "timestamp": time.time()})


def file_cache_middleware(func: Callable) -> Callable:
    """
    File-based cache middleware for functions that take a file path as first argument.
    """
    cache = FileCache()

    @wraps(func)
    def wrapper(file_path: str, *args, **kwargs):
        try:
            # Check cache first
            cached_content = cache.get(file_path)
            if cached_content is not None:
                return cached_content

            # If not in cache, call function and cache result
            result = func(file_path, *args, **kwargs)
            cache.set(file_path, result)
            return result
        except Exception as e:
            logger.error(
                "File cache error", error=str(e), func=func.__name__, file=file_path
            )
            return func(file_path, *args, **kwargs)

    return wrapper
