import json
import hashlib
from typing import Callable
from functools import wraps
import os
import time
import structlog

logger = structlog.get_logger()

def cache_middleware(func: Callable) -> Callable:
    """
    Cache middleware for Lumos functions.
    Currently disabled - passes through to original function.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper


def async_cache_middleware(func: Callable) -> Callable:
    """
    Cache middleware for async Lumos functions.
    Currently disabled - passes through to original function.
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        return await func(*args, **kwargs)
    return wrapper


def file_cache_middleware(func: Callable) -> Callable:
    """
    File-based cache middleware.
    Currently disabled - passes through to original function.
    """
    @wraps(func)
    def wrapper(file_path: str, *args, **kwargs):
        return func(file_path, *args, **kwargs)
    return wrapper
