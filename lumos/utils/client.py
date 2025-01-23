from typing import TypeVar
from pydantic import BaseModel
import httpx
import structlog
import os

logger = structlog.get_logger(__name__)

T = TypeVar("T", bound=BaseModel)


class LumosClient:
    def __init__(self, base_url: str, api_key: str):
        """Initialize Lumos client with base URL and API key."""
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.headers = {"X-API-Key": api_key}

        if not self.health_check():
            raise ConnectionError("Failed to connect to Lumos server")

        if api_key != os.getenv("LUMOS_API_KEY"):
            raise ValueError("Invalid API key")

        logger.info("Connected to Lumos server")

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

    def health_check(self) -> dict[str, str]:
        """Check if the Lumos server is healthy."""
        response = httpx.get(f"{self.base_url}/healthz", headers=self.headers)
        response.raise_for_status()
        return response.json()

    async def parse_book(self, pdf_path: str) -> dict:
        """Parse a PDF file and extract its sections and chunks."""
        async with httpx.AsyncClient() as client:
            # Open file in binary mode and create files dict
            with open(pdf_path, "rb") as f:
                files = {"file": ("file.pdf", f, "application/pdf")}
                response = await client.post(
                    f"{self.base_url}/book/parse-pdf",
                    headers=self.headers,
                    files=files,
                    timeout=30.0,
                )
                response.raise_for_status()
                ret = response.json()
                return ret["sections"], ret["chunks"]

    async def parse_file(self, file_path: str) -> dict:
        """Parse a non-PDF file and extract its sections and chunks.
        
        Args:
            file_path: Path to the file to parse. Supported formats:
                - .txt (text/plain)
                - .md (text/markdown)
                - .html/.htm (text/html) 
                - .docx (application/vnd.openxmlformats-officedocument.wordprocessingml.document)
                - .doc (application/msword)
                - .rtf (application/rtf)
        
        Returns:
            A tuple of (sections, chunks) where:
                - sections: List of section objects with metadata
                - chunks: List of content chunks with metadata
        """
        async with httpx.AsyncClient() as client:
            # Open file in binary mode and create files dict
            with open(file_path, "rb") as f:
                # Get the correct MIME type based on file extension
                file_ext = os.path.splitext(file_path)[1].lower()
                mime_types = {
                    '.txt': 'text/plain',
                    '.md': 'text/markdown',
                    '.html': 'text/html',
                    '.htm': 'text/html',
                    '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                    '.doc': 'application/msword',
                    '.rtf': 'application/rtf',
                }
                mime_type = mime_types.get(file_ext, 'application/octet-stream')
                
                files = {"file": (os.path.basename(file_path), f, mime_type)}
                response = await client.post(
                    f"{self.base_url}/book/parse-file",
                    headers=self.headers,
                    files=files,
                    timeout=30.0,
                )
                response.raise_for_status()
                ret = response.json()
                return ret["sections"], ret["chunks"]
