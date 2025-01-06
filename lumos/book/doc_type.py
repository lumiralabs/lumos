import base64
from pydantic import BaseModel, Field
from lumos import lumos
from .pdf_utils import extract_pdf_pages_as_images
import structlog

logger = structlog.get_logger(__name__)


def is_two_column_scientific_paper(pdf_path: str) -> bool:
    """
    Determine if a PDF document is a two-column scientific paper by analyzing its first two pages.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        bool: True if the document appears to be a two-column scientific paper
    """
    # Extract first two pages as images
    images_bytes = extract_pdf_pages_as_images(pdf_path, page_numbers=[1, 2])

    # Convert images to base64 for AI analysis
    message_content = []
    for image_data in images_bytes:
        encoded_image = base64.b64encode(image_data).decode("utf-8")
        message_content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"},
            }
        )

    class DocumentLayout(BaseModel):
        is_two_column: bool = Field(
            ..., description="True if the document appears to be in two-column format"
        )
        is_scientific_paper: bool = Field(
            ..., description="True if the document appears to be a scientific paper"
        )
        confidence: float = Field(..., description="Confidence score between 0-1")
        reasoning: str = Field(..., description="Explanation for the classification")

    # Use AI to analyze the layout
    result = lumos.call_ai(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a document layout analysis expert. "
                    "You need to determine if a document is a two-column scientific paper "
                    "by analyzing its first two pages. Look for these characteristics:\n"
                    "- Two distinct columns of text\n"
                    "- Academic/scientific formatting (title, authors, abstract)\n"
                    "- Presence of citations, equations, or technical figures\n"
                    "- Professional/academic appearance\n"
                    "- From Arxiv, IEEE, etc"
                    "Be conservative in your assessment - only return True if you're confident."
                ),
            },
            {
                "role": "user",
                "content": message_content,
            },
        ],
        model="gpt-4o-mini",
        response_format=DocumentLayout,
    )

    logger.info(f"Document layout analysis result: {result}")

    return (
        result.is_two_column and result.is_scientific_paper and result.confidence > 0.8
    )
