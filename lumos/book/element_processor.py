import copy
from typing import Any
from unstructured.chunking.title import chunk_by_title
from .models import Section
import structlog

logger = structlog.get_logger()


def get_elements_for_chapter(
    elements: list[Any] | list[dict], section: Section
) -> list:
    """
    Get elements for a chapter based on page boundaries.
    Chapter sections have clear page boundaries, so we can just filter by page numbers.
    """
    ret = []
    for e in elements:
        _e = e.to_dict() if not isinstance(e, dict) else e
        page_number = _e["metadata"]["page_number"]
        if page_number >= section.start_page and page_number <= section.end_page:
            ret.append(e)
    return ret


def normalize_text(text: str) -> str:
    return text.replace(" ", "").replace(".", "").replace("\t", "").strip().lower()


def is_title_match(text: str, title: str) -> bool:
    # If element starts with subsection title, it's definitely the start of this subsection
    _text = normalize_text(text)
    _title = normalize_text(title)
    return _text.startswith(_title)


def partition_section_elements(section: Section) -> Section:
    """
    Recursively partition elements into sections and their subsections based on section titles
    and page numbers. Elements are partitioned into the deepest matching subsection.
    """
    new_section = copy.deepcopy(section)

    # Base case: if no subsections, assign all elements to this section
    if not new_section.subsections:
        return new_section

    # Initialize partitions for each subsection
    elements: list[dict[Any, Any]] = section.elements
    partitions = {subsection.title: [] for subsection in new_section.subsections}
    unassigned_elements = []

    # Sort subsections by start page to handle overlapping ranges
    sorted_subsections = sorted(
        new_section.subsections, key=lambda x: x.start_page or float("inf")
    )

    # Track current subsection context
    current_subsection = None

    for element in elements:
        elem_dict = element.to_dict() if not isinstance(element, dict) else element
        page_number = elem_dict["metadata"]["page_number"]
        text = elem_dict["text"]

        # Find matching subsection based on page number
        matched = False
        for subsection in sorted_subsections:
            # Skip if subsection has no page numbers
            if subsection.start_page is None or subsection.end_page is None:
                continue

            # Check if element is within subsection's page range
            if subsection.start_page <= page_number <= subsection.end_page:
                if is_title_match(text, subsection.title):
                    current_subsection = subsection.title
                    logger.info(
                        "Found subsection",
                        current_subsection=current_subsection,
                        text=text[:30] + "..." if len(text) > 30 else text,
                        page_number=page_number,
                    )

                # If we have a current subsection and we're in its page range, assign to it
                if current_subsection == subsection.title:
                    partitions[current_subsection].append(element)
                    matched = True
                    break

        if not matched:
            unassigned_elements.append(element)

    # Recursively partition each subsection's elements
    for i in range(len(new_section.subsections)):
        subsection = new_section.subsections[i]
        subsection.elements = partitions[subsection.title]
        if subsection.subsections:
            new_section.subsections[i] = partition_section_elements(subsection)

    # Assign unassigned elements to the main section
    new_section.elements = unassigned_elements

    return new_section


def chunk_elements(elements: list) -> list:
    """Chunk elements using title-based chunking strategy."""
    return chunk_by_title(
        elements, max_characters=1000, new_after_n_chars=500, multipage_sections=True
    )


def add_chunks(section: Section) -> None:
    """Add chunks recursively to a section and its subsections."""
    if section.elements:
        section.chunks = chunk_elements(section.elements)
    if section.subsections:
        for subsection in section.subsections:
            add_chunks(subsection)


def get_leaf_sections(section: Section) -> list[tuple[str, str]]:
    """Get all leaf sections (those without subsections) and their elements."""
    results = []

    if section.elements and not section.subsections:
        # This is a leaf section - collect title and elements
        ele_str = "\n\n".join([element.text for element in section.elements])
        results.append((section.title, ele_str))

    # Recursively process subsections
    if section.subsections:
        for subsection in section.subsections:
            results.extend(get_leaf_sections(subsection))

    return results
