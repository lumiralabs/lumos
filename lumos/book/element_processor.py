import copy
from typing import Any
from unstructured.chunking.title import chunk_by_title
from .models import Section
import structlog
import re

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
    return text.replace(" ", "").replace(".", "").strip().lower()


def is_title_match(text: str, title: str) -> bool:
    # If element starts with subsection title, it's definitely the start of this subsection
    _text = normalize_text(text)
    _title = normalize_text(title)
    return _text.startswith(_title)


def get_section_number(title: str) -> tuple:
    """Extract section number from title like (1.2.3) or (1) or (1.2)"""
    match = re.match(r'\(([0-9.]+)\)', title)
    if match:
        return tuple(int(x) for x in match.group(1).split('.'))
    return (float('inf'),)  # For sections without numbers


def partition_section_elements(section: Section) -> Section:
    """
    Recursively partition elements into sections and their subsections based on section titles,
    page numbers, and section levels. Elements are partitioned into the appropriate level
    in the hierarchy (chapter -> section -> subsection).
    """
    new_section = copy.deepcopy(section)

    # Base case: if no subsections, assign all elements to this section
    if not new_section.subsections:
        return new_section

    # Initialize partitions for each subsection
    elements: list[dict[Any, Any]] = section.elements
    partitions = {subsection.title: [] for subsection in new_section.subsections}
    section_elements = []  # Elements that belong to this section level

    # Sort subsections by their section numbers to maintain TOC order
    sorted_subsections = sorted(
        new_section.subsections,
        key=lambda x: get_section_number(x.title)
    )

    # Track section context at each level
    current_contexts = {}  # level -> (section_title, section)
    
    for element in elements:
        elem_dict = element.to_dict() if not isinstance(element, dict) else element
        page_number = elem_dict["metadata"]["page_number"]
        text = elem_dict["text"]

        # Check if this element starts a new section at any level
        matched = False
        for subsection in sorted_subsections:
            if subsection.start_page is None or subsection.end_page is None:
                continue

            level = subsection.level if subsection.level is not None else 0
            
            # Check if element is within section's page range
            if subsection.start_page <= page_number <= subsection.end_page:
                # Check if this is a section title
                if is_title_match(text, subsection.title):
                    # Update context for this level
                    current_contexts[level] = (subsection.title, subsection)
                    
                    # Clear any deeper level contexts
                    current_contexts = {k: v for k, v in current_contexts.items() if k <= level}
                    
                    logger.info(
                        "Found section",
                        section_number='.'.join(str(x) for x in get_section_number(subsection.title)),
                        level=level,
                        current_section=subsection.title,
                        text=text[:30] + "..." if len(text) > 30 else text,
                        page_number=page_number,
                    )
                    
                    # Add title element to its section
                    partitions[subsection.title].append(element)
                    matched = True
                    break

                # Check if content belongs to current section at this level
                if level in current_contexts:
                    current_title, current_sec = current_contexts[level]
                    if current_title == subsection.title:
                        # Only assign to this section if we're not in a deeper level context
                        max_level = max(current_contexts.keys()) if current_contexts else -1
                        if level >= max_level:
                            partitions[current_title].append(element)
                            matched = True
                            break

        # If element wasn't matched and is within current section's range,
        # it belongs to the current section level
        if not matched and section.start_page <= page_number <= section.end_page:
            # Check if we're in any section context
            if not current_contexts or section.level >= max(current_contexts.keys()):
                section_elements.append(element)

    # Recursively partition each subsection's elements
    for i in range(len(new_section.subsections)):
        subsection = new_section.subsections[i]
        subsection.elements = partitions[subsection.title]
        if subsection.subsections:
            new_section.subsections[i] = partition_section_elements(subsection)

    # Assign section-level elements to this section
    new_section.elements = section_elements

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


def get_all_sections(section: Section) -> list[tuple[str, str]]:
    """Get all sections and their elements, including those with subsections."""
    results = []

    # Always include this section's elements, even if it has subsections
    if section.elements:
        ele_str = "\n\n".join([element.text for element in section.elements])
        results.append((section.title, ele_str))

    # Recursively process subsections
    if section.subsections:
        for subsection in section.subsections:
            results.extend(get_all_sections(subsection))

    return results
