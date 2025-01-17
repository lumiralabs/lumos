from typing import TypeVar
from pydantic import BaseModel
import httpx
from lumos.book import parser

T = TypeVar("T", bound=BaseModel)


class BookParser:
    def parse(self, pdf_path: str):
        return parser.parse(pdf_path)


class LumosClient:
    def __init__(self, base_url: str, api_key: str):
        """Initialize Lumos client with base URL and API key."""
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.headers = {"X-API-Key": api_key, "Content-Type": "application/json"}
        self.book = BookParser()

    async def call_ai_async(
        self,
        messages: list[dict[str, str]],
        response_format: type[T] | None = None,
        examples: list[tuple[str, T]] | None = None,
        model: str = "gpt-4o-mini",
    ) -> T | str:
        """
        Make an AI completion call to the Lumos server.

        Args:
            messages: list of chat messages to send
            response_format: A Pydantic model class defining the expected response structure
            examples: Optional list of (query, response) tuples for few-shot learning
            model: Model to use, defaults to "gpt-4o-mini"

        Returns:
            An instance of the response_format class or string containing the AI's response
        """
        # Prepare request payload
        payload = {
            "messages": messages,
            "model": model,
            "response_schema": None,
            "examples": None,
        }

        # Add schema if response_format is provided
        if response_format:
            payload["response_schema"] = response_format.model_json_schema()

        # Add examples if provided
        if examples:
            payload["examples"] = [
                (query, response.model_dump()) for query, response in examples
            ]

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/generate", headers=self.headers, json=payload
            )
            response.raise_for_status()
            data = response.json()

            if response_format:
                return response_format.model_validate(data)
            return data

    async def get_embedding(
        self, text: str | list[str], model: str = "text-embedding-3-small"
    ) -> list[float] | list[list[float]]:
        """Get embeddings for text using the Lumos server."""
        payload = {"inputs": text, "model": model}

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/embed", headers=self.headers, json=payload
            )
            response.raise_for_status()
            return response.json()

    async def health_check(self) -> dict[str, str]:
        """Check if the Lumos server is healthy."""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}/healthz", headers=self.headers
            )
            response.raise_for_status()
            return response.json()
