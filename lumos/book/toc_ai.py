import base64

import fitz  # PyMuPDF

from pydantic import BaseModel, Field
import structlog
import fire
from .visualizer import rich_view_toc_sections
from lumos import lumos
from .pdf_utils import extract_pdf_pages_as_images
from .models import TOC
from .toc_common import toc_list_to_toc_sections  # Import from new file


logger = structlog.get_logger()


# ---------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------


class TOC_LLM_SECTION(BaseModel):
    level: int
    title: str
    page: int | None = Field(
        ..., description="Page number of the section. It can be empty if not found."
    )


class TOC_LLM(BaseModel):
    sections: list[TOC_LLM_SECTION] = Field(
        ...,
        description="List of TOC sections. Pay attention to the indentation level, chapter parts and hierarchy.",
    )

    def to_list(self) -> list[list[int | str]]:
        """
        Convert each TOC line to a 3-element list [level, title, page].
        """
        return [
            [section.level, section.title, section.page] for section in self.sections
        ]


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
    pages_text: dict[int, str] = {}
    with open(pdf_path, "rb") as f:
        doc = fitz.open(f)
        for page_index, page in enumerate(doc, start=1):
            pages_text[page_index] = page.get_text()
    return pages_text


def extract_pdf_text_range(
    pdf_path: str, start_page: int, end_page: int | None = None
) -> str:
    """
    Extracts text for the pages from start_page to end_page (inclusive),
    returning it all as one concatenated string. (1-based)
    """
    logger.debug("extracting_pdf_text_range", range=(start_page, end_page))
    all_pages_text = extract_all_pdf_text(pdf_path)
    text_segments = []
    if end_page is None:
        if start_page in all_pages_text:
            text_segments.append(all_pages_text[start_page])
    else:
        for page_num in range(start_page, end_page + 1):
            if page_num in all_pages_text:
                text_segments.append(all_pages_text[page_num])
    return "\n".join(text_segments)


def search_for_title(title: str, doc_pages: list[str], start_offset: int = 0) -> int:
    """Search for a title in the document pages, starting from start_offset."""
    for page_num, page_text in enumerate(doc_pages[start_offset:], start=start_offset):
        if title in page_text:
            return page_num
    return None


# ---------------------------------------------------------------------
# TOC Extraction
# ---------------------------------------------------------------------


def sanitize_toc_list(toc_list: list[list]) -> list[list]:
    """
    Fixes heirarchy parsing errors and removes unnecessary sections.
    Good for markdown section title parsing
    """

    toc_list_str = "\n".join(
        [f"({idx}) L-{i[0]} > {i[1]}" for idx, i in enumerate(toc_list, 1)]
    )

    class _TOC_ITEM(BaseModel):
        level: int
        title: str

    class TOC_LIST(BaseModel):
        sections: list[_TOC_ITEM] = Field(
            ...,
            description="List of TOC sections. Pay attention to the indentation level, chapter parts and hierarchy.",
        )

        def to_list(self) -> list[list[int | str]]:
            """
            Convert each TOC line to a 3-element list [level, title, page = None].
            """
            return [[section.level, section.title, None] for section in self.sections]

    toc_lis = lumos.call_ai(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant that is assisting in cleaning the extracted Table of Contents from a arxiv document. "
                    "We want to correct for some parsing errors in the levels and heirarchy. "
                    "Main sections of the material should be at level 1, and sub-sections should be at level 2, etc. "
                    "Often times there are some errors in the parsing, and sometimes the title of the book is at level 1 and the most important parts are at a below level. "
                    "Correct for this - remove these tasks and only keep the main ones, remove appendix, only keep th "
                    "The top level section is 1, and the next level is 2, etc. "
                    "Follow these rules:\n"
                    "1. Preserve the exact titles as they appear in the text\n"
                    "2. Keep the same page numbers as in the original\n"
                    "3. Maintain the same section numbering and hierarchy\n"
                    "4. Include all front matter sections (Cover, Copyright, etc.)\n"
                    "5. Do not modify or normalize the text case\n"
                    "6. Keep the exact same order of sections"
                    "7. Ignore References, Bibliography, Acknowledgements, Appendix, etc."
                ),
            },
            {
                "role": "user",
                "content": toc_list_str,
            },
        ],
        model="gpt-4o",
        response_format=TOC_LIST,
    )

    return toc_lis.to_list()


def extract_toc_llm(
    pdf_path: str, page_range: tuple[int, int | None] | None = None
) -> TOC_LLM:
    """
    Extract table of contents from a PDF file.

    Args:
        pdf_path: Path to the PDF file
        toc_pages: List of 1-based page numbers containing the table of contents

    Returns:
        List of [level, title, page] entries representing the table of contents

    TODO:
    - hallucinations of page numbers for parts which break the page number boundries
    - indentation issue breaks the heirarchy, but atleast all leaf sections are mostly valid
    """
    logger.info("extracting_toc", pdf_path=pdf_path, page_range=page_range)

    # Extract text from TOC pages
    start_page, end_page = page_range
    toc_raw_text = extract_pdf_text_range(pdf_path, start_page, end_page)
    images_bytes = extract_pdf_pages_as_images(pdf_path, [start_page, end_page])

    message_content = []
    message_content.append(
        {
            "type": "text",
            "text": f"Here is the extracted text from pages {start_page} to {end_page}:\n{toc_raw_text}",
        }
    )
    for idx, image_data in enumerate(images_bytes, start=start_page):
        encoded_image = base64.b64encode(image_data).decode("utf-8")
        message_content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"},
            }
        )

    # Call GPT to extract structured TOC
    toc_llm = lumos.call_ai(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant that can extract a Table of Contents. "
                    "You have BOTH the extracted text and direct images for the same pages. "
                    "Use them together to produce a final, accurate TOC. "
                    "The top level section is 1, and the next level is 2, etc. "
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
        response_format=TOC_LLM,
    )

    logger.info("extracted_toc", num_entries=len(toc_llm.sections))
    return toc_llm


def get_offset(toc: TOC_LLM, doc_pages: list[str], start_offset: int = 0) -> int:
    """Calculate offset between reported page numbers and actual PDF page numbers.

    Args:
        list_title_pageno: List of [title, page] entries
        doc_pages: List of page text content from PDF
        start_offset: Offset to start searching from (cuz after the contents page)

    Returns:
        int: Offset between reported and actual page numbers
    """
    logger.info(
        "calculating_page_offset",
        num_entries=len(toc.sections),
        start_offset=start_offset,
    )
    offsets = []

    for section in toc.sections:
        _, title, page = section.level, section.title, section.page
        logger.debug("searching_for_title", title=title, predicted_page=page)
        # Search through pages for title
        if page is None:
            logger.warning("None page - skipping", title=title)
            continue
        for page_num, page_text in enumerate(
            doc_pages[start_offset:], start=start_offset
        ):
            if title in page_text:
                # Found title, calculate offset
                actual_page = page_num
                predicted_page = page - 1  # Convert to 0-based
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
            logger.warning("title_not_found", title=title)

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
) -> tuple[list[list[int | str]], tuple[int, int]]:
    if toc_page_range is None:
        toc_pages = detect_toc_pages(pdf_path)
        toc_page_range = (min(toc_pages), max(toc_pages))
        if not toc_pages:
            raise ValueError("No TOC pages found in the first 10 pages of the PDF.")

    toc_llm = extract_toc_llm(pdf_path, toc_page_range)
    toc_list = toc_llm.to_list()
    with fitz.open(pdf_path) as doc:
        total_pages = len(doc)
    sections = toc_list_to_toc_sections(toc_list, total_pages)

    rich_view_toc_sections(sections)
    return toc_list, toc_page_range


def extract_toc(pdf_path: str) -> TOC:
    """
    Extract TOC from a PDF file using AI + offset calculation
    """
    toc_list, toc_page_range = extract_toc_ai(pdf_path)
    with fitz.open(pdf_path) as doc:
        pages_str = [p.get_text() for p in doc.pages()]
        total_pages = len(doc)
    logger.info("calculating_page_offset", start_offset=max(toc_page_range))
    offset = get_offset(toc_list, pages_str, start_offset=max(toc_page_range))

    toc_list_adjusted = [
        [level, title, page + offset] for level, title, page in toc_list
    ]
    sections = toc_list_to_toc_sections(toc_list_adjusted, total_pages)
    return TOC(sections=sections, total_pages=total_pages)


def extract_text_image_pairs(
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


def detect_toc_pages(pdf_path: str, max_pages: int = 15) -> list[int]:
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
    text_image_pairs = extract_text_image_pairs(pdf_path, 1, max_pages)

    # Build message content for AI
    message_content = []
    for page_num, (text, image) in enumerate(text_image_pairs, start=1):
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


def detect_page_for_title(
    title: str, pdf_path: str, page_range: tuple[int, int]
) -> int | None:
    text_image_pairs = extract_text_image_pairs(pdf_path, page_range[0], page_range[1])
    """
    Given a title, find the page in the text_image_pairs that contains the title.
    This is used to detect chapter headings.
    """

    class PageMatch(BaseModel):
        match: bool = Field(
            ..., description="True if the title appears as a heading, False otherwise"
        )
        reason: str = Field(..., description="Reason for the match or mismatch")
        confidence: float = Field(
            ...,
            description="Confidence score between 0-1 that this is a section heading",
        )

    logger.info("searching_for_title", title=title)

    # Build message content for AI, starting from the end
    responses = []
    for text, image in reversed(text_image_pairs):
        # Add text block
        message_content = []
        message_content.append({"type": "text", "text": text})

        # Add image block if available
        if image:
            encoded_image = base64.b64encode(image).decode("utf-8")
            message_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"},
                }
            )

        # First message to set context
        result = lumos.call_ai(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant that can identify section headings in a book. "
                        "You will be given a page of a book and you need to decide if the title appears as a heading. "
                        "Look for these characteristics of a heading:\n"
                        "- Appears prominently at the top of the page\n"
                        "- Larger or different font than body text\n"
                        "- May have decorative elements or spacing around it\n"
                        "- Starts a new section or chapter"
                        " Be very conservative in your answer. We are testing this on a few pages so be strict. If you are not sure, return False."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Find the page where this title appears as a heading: {title}\n\nHere are the pages to check: ",
                },
                {"role": "user", "content": message_content},
            ],
            model="gpt-4o",
            response_format=PageMatch,
        )

        responses.append(result)

    class PageMatch2(BaseModel):
        page: int = Field(
            ..., description="Page number where the title appears as a heading"
        )
        reason: str = Field(..., description="Reason for the match or mismatch")

    query_str = "\n".join(
        [f"Page ({i}): {r.reason}" for i, r in enumerate(responses, 1)]
    )
    print(query_str)
    ret = lumos.call_ai(
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that can identify section headings in a book. You will be given a list of responses from an AI and you need to determine the page number where the title appears as a heading. Remember that the indexing starts from 1.",
            },
            {"role": "user", "content": f"Here are the responses: {responses}"},
        ],
        model="gpt-4o",
        response_format=PageMatch2,
    )
    print(ret.model_dump_json())

    return page_range[1] - ret.page + 1


# ---------------------------------------------------------------------
# CLI Commands
# ---------------------------------------------------------------------


class CLI:
    """CLI commands for working with book TOCs"""

    def detect(self, pdf_path: str, max_pages: int = 15) -> list[int]:
        pages = detect_toc_pages(pdf_path, max_pages)
        print(pages)

    def extract(
        self, pdf_path: str, start_page: int | None = None, end_page: int | None = None
    ) -> list[list[int | str]]:
        toc_page_range = None
        if start_page and end_page:
            toc_page_range = (start_page, end_page)

        extract_toc_ai(pdf_path, toc_page_range)
        # sections = _get_section_hierarchy(toc_list, total_pages)
        # rich_view_toc_sections(sections)

    def offset(
        self, pdf_path: str, start_page: int, end_page: int
    ) -> list[list[int | str]]:
        toc_llm = extract_toc_llm(pdf_path, [start_page, end_page])
        with fitz.open(pdf_path) as doc:
            pages_str = [p.get_text() for p in doc.pages()]
        return get_offset(toc_llm, pages_str, start_offset=end_page)


if __name__ == "__main__":
    fire.Fire(CLI)
