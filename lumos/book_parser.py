import os
import structlog
from lumos.book.toc import extract_toc_from_pdf_metadata, toc_list_to_toc_sections
from lumos.book.models import TOC, Section
from lumos.book.visualizer import rich_view_toc_sections
import fitz  # PyMuPDF

logger = structlog.get_logger(__name__)

def filter_sections_by_level(sections: list[Section], max_level: int) -> list[Section]:
    """
    Filters sections to include only those within the specified depth level.
    
    Args:
        sections (list[Section]): The list of structured TOC sections.
        max_level (int): The maximum depth level to keep.

    Returns:
        list[Section]: Filtered list of sections.
    """
    filtered_sections = []

    for section in sections:
        # Extract the top-level number from "1", "1.1", etc.
        section_level = len(section.level.split('.'))
        
        if section_level <= max_level:
            # Recursively filter subsections
            if section.subsections:
                section.subsections = filter_sections_by_level(section.subsections, max_level)
            filtered_sections.append(section)

    return filtered_sections

def toc(pdf_path: str, level: int = 1) -> TOC:
    """
    Extract and display a structured Table of Contents (TOC) from a PDF.

    Args:
        pdf_path (str): Path to the PDF file.
        level (int, optional): Maximum TOC depth level to display. Defaults to 1.

    Returns:
        TOC object: Structured TOC representation.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"File not found: {pdf_path}")

    logger.info("Extracting TOC from PDF metadata", pdf_path=pdf_path)
    toc_list = extract_toc_from_pdf_metadata(pdf_path)

    if not toc_list:
        raise ValueError(f"Could not extract table of contents from {pdf_path}")

    # Get total pages in PDF
    with fitz.open(pdf_path) as doc:
        total_pages = len(doc)

    # Convert TOC list to structured TOC sections
    sections = toc_list_to_toc_sections(toc_list, total_pages)

    # Apply level filtering
    filtered_sections = filter_sections_by_level(sections, level)

    # Create TOC object
    toc_obj = TOC(sections=filtered_sections)

    # Display TOC using rich text visualization
    rich_view_toc_sections(toc_obj.sections)

    return toc_obj  # Return the TOC object for further processing if needed
