from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, create_model, HttpUrl
from typing import Literal, Any, Callable
import lumos
from functools import wraps
import tempfile
import os
import requests
from fastapi import UploadFile, File
from ..book.parser import from_pdf_path
import structlog

logger = structlog.get_logger(__name__)
app = FastAPI(title="Lumos API")

LUMOS_API_KEY = os.getenv("LUMOS_API_KEY")


def require_api_key(func: Callable):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        request: Request = kwargs.get("request") or args[0]
        api_key = request.headers.get("X-API-Key")
        if not api_key:
            raise HTTPException(status_code=401, detail="API key is missing")
        if api_key != LUMOS_API_KEY:
            raise HTTPException(status_code=401, detail="Invalid API key")
        return await func(*args, **kwargs)

    return wrapper


class ChatMessage(BaseModel):
    role: str = Literal["system", "user", "assistant", "developer"]
    content: str


class AIRequest(BaseModel):
    messages: list[ChatMessage]
    response_schema: dict[str, Any] | None  # JSONschema
    examples: list[tuple[str, dict[str, Any]]] | None = None
    model: str | None = "gpt-4o-mini"


class EmbedRequest(BaseModel):
    inputs: str | list[str]
    model: str | None = "text-embedding-3-small"


class PDFRequest(BaseModel):
    url: HttpUrl | None = None


def _json_schema_to_pydantic_types(
    schema: dict[str, Any],
) -> dict[str, tuple[type, Any]]:
    """Convert JSON schema types to Python/Pydantic types"""
    type_mapping = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "array": list,
        "object": dict,
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


@app.post("/generate")
@require_api_key
async def create_chat_completion(request: Request, ai_request: AIRequest):
    """
    Examples can only be used if response_schema is provided, and are in json format
    """
    try:
        ResponseModel = None
        formatted_examples = None

        # Convert JSON schema to Pydantic field types
        if ai_request.response_schema:
            field_types = _json_schema_to_pydantic_types(ai_request.response_schema)
            ResponseModel = create_model("DynamicResponseModel", **field_types)

        if ai_request.examples:
            formatted_examples = [
                (query, ResponseModel.model_validate(response))
                for query, response in ai_request.examples
            ]

        # Convert messages to dict format
        messages = [msg.model_dump() for msg in ai_request.messages]

        # Call the AI function
        result = await lumos.call_ai_async(
            messages=messages,
            response_format=ResponseModel,
            examples=formatted_examples,
            model=ai_request.model,
        )

        return result.model_dump()

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/embed")
@require_api_key
async def embed(request: Request, embed_request: EmbedRequest):
    return lumos.get_embedding(embed_request.inputs, embed_request.model)


@app.get("/healthz")
async def health_check(request: Request):
    return {"status": "healthy"}


@app.get("/")
async def root(request: Request):
    return {"message": "Lumos API"}


@app.post("/book/parse-pdf")
@require_api_key
async def process_pdf(
    request: Request,
    pdf_request: PDFRequest | None = None,
    file: UploadFile | None = File(None),
):
    """Process a PDF file from either a URL or uploaded file."""
    # Log request details
    logger.info(
        f"PDF processing request - PDF URL: {pdf_request.url if pdf_request else None}, File: {file.filename if file else None}"
    )

    if not pdf_request and not file:
        logger.error("No PDF URL or file provided")
        raise HTTPException(
            status_code=400, detail="Either a PDF URL or file upload must be provided"
        )

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            if pdf_request and pdf_request.url:
                # Download from URL
                logger.info(f"Downloading PDF from URL: {pdf_request.url}")
                try:
                    response = requests.get(pdf_request.url)
                    response.raise_for_status()
                    tmp_file.write(response.content)
                except requests.RequestException as e:
                    logger.error(
                        f"Failed to download PDF from {pdf_request.url}: {str(e)}"
                    )
                    raise HTTPException(
                        status_code=400, detail=f"Failed to download PDF: {str(e)}"
                    )
            elif file:
                # Handle uploaded file
                logger.info(f"Processing uploaded file: {file.filename}")
                try:
                    content = await file.read()
                    tmp_file.write(content)
                except Exception as e:
                    logger.error(
                        f"Failed to read uploaded file {file.filename}: {str(e)}"
                    )
                    raise HTTPException(
                        status_code=400,
                        detail=f"Failed to read uploaded file: {str(e)}",
                    )

            tmp_file.flush()

            # Process the PDF
            try:
                logger.info(f"Processing PDF file: {tmp_file.name}")
                book = from_pdf_path(tmp_file.name)
                sections = book.flatten_sections(only_leaf=False)
                raw_chunks = book.flatten_chunks()

                logger.info(
                    f"Successfully processed PDF with {len(sections)} sections and {len(raw_chunks)} chunks"
                )
                return {"sections": sections, "chunks": raw_chunks}
            except Exception as e:
                logger.error(f"Failed to process PDF content: {str(e)}")
                raise HTTPException(
                    status_code=500, detail=f"Failed to process PDF content: {str(e)}"
                )
            finally:
                # Clean up temp file
                logger.debug(f"Cleaning up temporary file: {tmp_file.name}")
                os.unlink(tmp_file.name)

    except Exception as e:
        logger.error(f"Unexpected error processing PDF: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Unexpected error processing PDF: {str(e)}"
        )
