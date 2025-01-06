import json
import hashlib
from typing import Callable
from functools import wraps
import os
import time


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


class FileCache:
    def __init__(self, cache_dir: str = ".lumos_file_cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_index_file = os.path.join(cache_dir, "index.json")

        # Load existing cache index or create new one
        try:
            with open(self.cache_index_file, "r") as f:
                self.cache_index = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            self.cache_index = {}

    def _get_file_hash(self, file_path: str) -> str:
        """Calculate SHA-256 hash of a file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            # Read the file in chunks to handle large files
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def get(self, file_path: str) -> tuple[str, str] | None:
        """
        Get cached content for a file if it exists.
        Returns (pdf_id, markdown_path) if cached, None otherwise.
        """
        file_hash = self._get_file_hash(file_path)
        cache_entry = self.cache_index.get(file_hash)
        if cache_entry is None:
            return None

        markdown_path = os.path.join(self.cache_dir, f"{cache_entry['pdf_id']}.md")
        if not os.path.exists(markdown_path):
            return None

        return cache_entry["pdf_id"], markdown_path

    def set(self, file_path: str, pdf_id: str, markdown_content: str) -> str:
        """
        Cache the markdown content for a file.
        Returns the path to the cached markdown file.
        """
        file_hash = self._get_file_hash(file_path)
        markdown_path = os.path.join(self.cache_dir, f"{pdf_id}.md")

        # Save the markdown content
        with open(markdown_path, "w") as f:
            f.write(markdown_content)

        # Update the cache index
        self.cache_index[file_hash] = {"pdf_id": pdf_id, "timestamp": time.time()}

        # Save the updated index
        with open(self.cache_index_file, "w") as f:
            json.dump(self.cache_index, f)

        return markdown_path


def file_cache_middleware(func: Callable) -> Callable:
    """
    File-based cache middleware for functions that take a file path as first argument.
    """
    cache = FileCache()

    @wraps(func)
    def wrapper(file_path: str, *args, **kwargs):
        # Check cache first
        cache_result = cache.get(file_path)
        if cache_result is not None:
            pdf_id, markdown_path = cache_result
            with open(markdown_path, "r") as f:
                return f.read()

        # If not in cache, call function and cache result
        result = func(file_path, *args, **kwargs)

        # Generate a unique ID for this conversion
        pdf_id = hashlib.sha256(f"{file_path}_{time.time()}".encode()).hexdigest()[:12]
        cache.set(file_path, pdf_id, result)

        return result

    return wrapper
