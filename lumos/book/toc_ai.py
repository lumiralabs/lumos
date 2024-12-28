import io
import base64

import fitz  # PyMuPDF
from pdf2image import convert_from_path
from pydantic import BaseModel, Field
import structlog

from lumos import lumos

logger = structlog.get_logger()
logger = logger.bind(module="[toc_ai]")


# ---------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------


class TOCLine(BaseModel):
    level: int
    title: str
    page: int


class TOC_JSON(BaseModel):
    lines: list[TOCLine]

    def to_list(self) -> list[list[int | str]]:
        """
        Convert each TOC line to a 3-element list [level, title, page].
        """
        return [[line.level, line.title, line.page] for line in self.lines]


# ---------------------------------------------------------------------
# PDF Utilities (All 1-based)
# ---------------------------------------------------------------------


def extract_all_pdf_text(pdf_path: str) -> dict[int, str]:
    """
    Returns a dictionary of all pages' text, keyed by 1-based page number.

    Example usage:
      pages_text = extract_all_pdf_text("file.pdf")
      # pages_text[1] -> text of the first page
      # pages_text[2] -> text of the second page, etc.
    """
    logger.info("extracting_all_pdf_text", pdf_path=pdf_path)
    pages_text: dict[int, str] = {}
    with open(pdf_path, "rb") as f:
        doc = fitz.open(f)
        for page_index, page in enumerate(doc, start=1):
            pages_text[page_index] = page.get_text()
    logger.debug("extracted_pdf_text", num_pages=len(pages_text))
    return pages_text


def extract_pdf_text_range(pdf_path: str, start_page: int, end_page: int) -> str:
    """
    Extracts text for the pages from start_page to end_page (inclusive),
    returning it all as one concatenated string. (1-based)
    """
    logger.debug("extracting_pdf_text_range", start_page=start_page, end_page=end_page)
    all_pages_text = extract_all_pdf_text(pdf_path)
    text_segments = []
    for page_num in range(start_page, end_page + 1):
        if page_num in all_pages_text:
            text_segments.append(all_pages_text[page_num])
    return "\n".join(text_segments)


def extract_pdf_pages_as_images(pdf_path: str, page_numbers: list[int]) -> list[bytes]:
    """
    Extracts the given 1-based page numbers from a PDF as JPG images.

    :param pdf_path: The path to the PDF file.
    :param page_numbers: 1-based page numbers to extract.
    :return: A list of bytes (each representing a single page image in JPG format).
    """
    logger.debug("extracting_pdf_pages_as_images", page_numbers=page_numbers)
    images_as_bytes = []
    for page_number in page_numbers:
        # pdf2image expects 1-based pages, so we can use page_number directly
        pages = convert_from_path(
            pdf_path,
            dpi=100,
            first_page=page_number,
            last_page=page_number,
        )
        if pages:
            # Typically one image per page.
            image = pages[0]
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG")
            buffer.seek(0)
            images_as_bytes.append(buffer.read())
    logger.debug("extracted_pdf_images", num_images=len(images_as_bytes))
    return images_as_bytes


def extract_toc(pdf_path: str, toc_pages: list[int]) -> list[list[int | str]]:
    """
    Extract table of contents from a PDF file.

    Args:
        pdf_path: Path to the PDF file
        toc_pages: List of 1-based page numbers containing the table of contents

    Returns:
        List of [level, title, page] entries representing the table of contents
    """
    logger.info("extracting_toc", pdf_path=pdf_path, toc_pages=toc_pages)

    # Extract text from TOC pages
    start_page = min(toc_pages)
    end_page = max(toc_pages)
    toc_raw_text = extract_pdf_text_range(pdf_path, start_page, end_page)

    # Extract images for TOC pages
    images_bytes = extract_pdf_pages_as_images(pdf_path, toc_pages)

    # Build message content with text and images
    message_content = []

    # Add text block
    message_content.append(
        {
            "type": "text",
            "text": f"Here is the extracted text from pages {start_page} to {end_page}:\n{toc_raw_text}",
        }
    )

    # Add image blocks
    for idx, image_data in enumerate(images_bytes, start=start_page):
        encoded_image = base64.b64encode(image_data).decode("utf-8")
        message_content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"},
            }
        )

    # Call GPT to extract structured TOC
    final_toc = lumos.call_ai(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant that can extract a Table of Contents. "
                    "You have BOTH the extracted text and direct images for the same pages. "
                    "Use them together to produce a final, accurate TOC. "
                    "Follow these rules:\n"
                    "1. Preserve the exact titles as they appear in the text\n"
                    "2. Keep the same page numbers as in the original\n"
                    "3. Maintain the same section numbering and hierarchy\n"
                    "4. Include all front matter sections (Cover, Copyright, etc.)\n"
                    "5. Do not modify or normalize the text case\n"
                    "6. Keep the exact same order of sections"
                ),
            },
            {
                "role": "user",
                "content": (
                    "Please generate a Table of Contents from the following text and images. "
                    "Preserve indentation hierarchy. Ignore leading numerals that are not part "
                    "of the actual title. Output the final TOC in the specified JSON format."
                ),
            },
            {
                "role": "user",
                "content": message_content,
            },
        ],
        model="gpt-4o",
        response_format=TOC_JSON,
    )

    logger.info("extracted_toc", num_entries=len(final_toc.lines))
    return final_toc


def get_offset(
    top_level_toc: TOC_JSON, doc_pages: list[str], start_offset: int = 0
) -> int:
    """Calculate offset between reported page numbers and actual PDF page numbers.

    Args:
        top_level_toc: List of TOCLine objects containing top level entries
        doc_pages: List of page text content from PDF
        start_offset: Offset to start searching from (cuz after the contents page)

    Returns:
        int: Offset between reported and actual page numbers
    """
    logger.info(
        "calculating_page_offset",
        num_entries=len(top_level_toc),
        start_offset=start_offset,
    )
    offsets = []

    for toc_entry in top_level_toc:
        logger.debug(
            "searching_for_title", title=toc_entry.title, predicted_page=toc_entry.page
        )
        # Search through pages for title
        for page_num, page_text in enumerate(
            doc_pages[start_offset:], start=start_offset
        ):
            if toc_entry.title in page_text:
                # Found title, calculate offset
                actual_page = page_num
                predicted_page = toc_entry.page - 1  # Convert to 0-based
                offset = actual_page - predicted_page
                offsets.append(offset)
                logger.debug(
                    "found_title",
                    actual_page=actual_page,
                    pdf_page=actual_page + 1,
                    offset=offset,
                    predicted_page=predicted_page,
                )
                break
        else:
            logger.warning("title_not_found", title=toc_entry.title)

    if not offsets:
        logger.error("no_offsets_found")
        raise ValueError("Could not find any top level titles in document")

    most_common = max(set(offsets), key=offsets.count)
    logger.info(
        "offset_calculation_complete",
        all_offsets=offsets,
        most_common_offset=most_common,
    )

    return most_common


def extract_toc_ai(
    pdf_path: str, toc_page_range: tuple[int, int] | None = None
) -> list[list[int | str]]:
    # toc_page_range is (start_page, end_page) inclusive as 1-based
    logger.info("starting_toc_extraction", pdf_path=pdf_path)

    # First try to get TOC from PDF metadata
    with fitz.open(pdf_path) as doc:
        pages_str = [p.get_text() for p in doc.pages()]

    # If toc_page_range not provided, detect TOC pages in first 10 pages
    if toc_page_range is None:
        toc_pages = detect_toc_pages(pdf_path)
        if not toc_pages:
            raise ValueError("No TOC pages found in the first 10 pages of the PDF.")
    else:
        start_page, end_page = toc_page_range
        toc_pages = list(range(start_page, end_page + 1))

    # Extract TOC using AI
    toc = extract_toc(pdf_path, toc_pages)
    offset = get_offset(toc.lines, pages_str, start_offset=max(toc_pages))

    for entry in toc.lines:
        entry.page += offset
    return toc.to_list()


def extract_page_content_range(
    pdf_path: str, start_page: int, end_page: int
) -> list[tuple[str, bytes | None]]:
    """
    Extract text and image content for a range of pages.

    Args:
        pdf_path: Path to the PDF file
        start_page: Start page number (1-based)
        end_page: End page number (1-based)

    Returns:
        List of tuples (text, image_bytes) for each page in range
    """
    logger.debug(
        "extracting_page_content_range", start_page=start_page, end_page=end_page
    )

    # Get text content
    text_content = extract_pdf_text_range(pdf_path, start_page, end_page)
    pages_text = text_content.split("\n")

    # Get image content
    page_numbers = list(range(start_page, end_page + 1))
    images_bytes = extract_pdf_pages_as_images(pdf_path, page_numbers)

    # Pair text with images
    content = []
    for text, image in zip(pages_text, images_bytes):
        content.append((text, image))

    return content


def detect_toc_pages(pdf_path: str, max_pages: int = 10) -> list[int]:
    """
    Detect which pages in the first max_pages contain table of contents.

    Args:
        pdf_path: Path to the PDF file
        max_pages: Maximum number of pages to check (default: 10)

    Returns:
        List of 1-based page numbers that likely contain table of contents
    """
    logger.info("detecting_toc_pages", pdf_path=pdf_path, max_pages=max_pages)

    # Extract content from first max_pages
    content = extract_page_content_range(pdf_path, 1, max_pages)

    # Build message content for AI
    message_content = []
    for page_num, (text, image) in enumerate(content, start=1):
        # Add text block
        message_content.append({"type": "text", "text": f"Page {page_num}:\n{text}"})

        # Add image block if available
        if image:
            encoded_image = base64.b64encode(image).decode("utf-8")
            message_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"},
                }
            )

    # Call GPT to identify TOC pages
    class TOCPages(BaseModel):
        pages: list[int] = Field(
            ..., description="List of 1-based page numbers containing table of contents"
        )

    toc_pages = lumos.call_ai(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant that can identify pages containing a table of contents in a book. "
                    "You have both the text and images for each page. "
                    "Look for pages that contain structured lists of chapters/sections with page numbers. "
                    "Ignore pages that just mention 'Contents' or 'Table of Contents' but don't actually contain the TOC. "
                    "Return only the page numbers that actually contain TOC entries."
                ),
            },
            {
                "role": "user",
                "content": message_content,
            },
        ],
        model="gpt-4o",
        response_format=TOCPages,
    )

    logger.info("detected_toc_pages", pages=toc_pages.pages)
    return toc_pages.pages
