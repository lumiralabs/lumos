import copy
from typing import Any
from unstructured.chunking.title import chunk_by_title
from .models import Section


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


def partition_elements(section: Section) -> Section:
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

    # Track current section context
    current_subsection = None
    for element in elements:
        elem_dict = element.to_dict() if not isinstance(element, dict) else element
        page_number = elem_dict["metadata"]["page_number"]

        # Find matching subsection based on page number and title
        for subsection in new_section.subsections:
            if (
                elem_dict["text"].startswith(subsection.title)
                and page_number == subsection.start_page
            ):
                current_subsection = subsection.title
                break

        if current_subsection:
            partitions[current_subsection].append(element)
        else:
            unassigned_elements.append(element)

    # Recursively partition each subsection's elements
    for i in range(len(new_section.subsections)):
        subsection = new_section.subsections[i]
        subsection.elements = partitions[subsection.title]
        if subsection.subsections:
            new_section.subsections[i] = partition_elements(subsection)

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
