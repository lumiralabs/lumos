import re
import fitz
from pydantic import BaseModel, Field
from .models import Section, TOC
from lumos import lumos
from typing import Literal
import structlog
from .toc_ai import extract_toc_ai

logger = structlog.get_logger(__name__)


def extract_chapters_by_pattern(sections: list[Section]) -> list[Section]:
    """Extract chapters from sections using regex pattern matching."""
    chapter_pattern = re.compile(r"^chapter \d+(?!.*appendix).*$", re.IGNORECASE)
    return [s for s in sections if chapter_pattern.match(s.title)]


def _toc_to_str(sections: list[Section], parent_idx: str = "", level: int = 0) -> str:
    """Convert sections to a string representation for AI processing recursively."""
    result = ""
    for i, section in enumerate(sections):
        if level == 0:
            # Top level sections
            result += f"({i}) {section.title} (Pages: {section.start_page}-{section.end_page})\n"
        else:
            # Non-top level sections with indent and dash
            indent = "  " * (level + 1)  # Add one more indent level
            result += f"{indent}- {section.title} (Pages: {section.start_page}-{section.end_page})\n"
        if section.subsections:
            result += _toc_to_str(section.subsections, f"{i}.", level + 1)
    return result


def extract_chapters_ai(sections: list[Section]) -> list[Section]:
    """Use AI to identify main chapters when pattern matching fails."""
    toc_str = _toc_to_str(sections)

    class BookTopLevel(BaseModel):
        type: Literal["chapter", "part"]
        indices: list[int] = Field(..., description="List of part or chapter numbers")

    ret = lumos.call_ai(
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that can identify chapter numbers from a table of contents. "
                "Given this table of contents, identify the line numbers (in parentheses) that contain actual numbered chapters (e.g. (1), (2), etc). "
                "Ignore sections like 'Table of Contents', 'Index', 'Acknowledgements', Appendices, etc. "
                "We want the most important parts and chapters relevant for study. "
                "Return only a list of indices for the top level of the TOC. That is enough for selecting the chapters. "
                "Sometimes the top level TOC has Part 1, Part 2, inside which are the actual chapters. "
                "In that case return the indices of the parts. For parts, focus on the most relevant parts, ignore Bonus content. "
                "This will do the job for filtering out the chapters.",
            },
            {"role": "user", "content": toc_str},
        ],
        model="gpt-4o",
        response_format=BookTopLevel,
    )
    return [
        Section(
            level=s.level,
            title=s.title,
            start_page=s.start_page,
            end_page=s.end_page,
            subsections=s.subsections,
            type=ret.type,
        )
        for i, s in enumerate(sections)
        if i in ret.indices
    ]


def extract_chapters(sections: list[Section]) -> list[Section]:
    """Extract main chapters from sections using pattern matching first, then AI if needed."""
    chapters = extract_chapters_by_pattern(sections)
    if not chapters:
        chapters = extract_chapters_ai(sections)
    return chapters


def _get_section_hierarchy(toc: list[list], total_pages: int) -> list[Section]:
    """Build a hierarchical structure of sections from the TOC."""

    def recursive_parse(level, toc, index, parent_end_page, parent_level=""):
        sections = []
        section_num = 1
        while index < len(toc):
            curr_level, title, page = toc[index]
            if curr_level < level:
                break

            end_page = parent_end_page
            for next_index in range(index + 1, len(toc)):
                next_level, _, next_page = toc[next_index]
                if next_level <= curr_level:
                    end_page = next_page - 1
                    break

            current_level = (
                f"{parent_level}{section_num}" if parent_level else str(section_num)
            )
            subsection, next_index = recursive_parse(
                curr_level + 1, toc, index + 1, end_page, f"{current_level}."
            )

            sections.append(
                Section(
                    level=current_level,
                    title=title,
                    start_page=page,
                    end_page=end_page,
                    subsections=subsection or None,
                )
            )
            section_num += 1
            index = next_index
        return sections, index

    top_level_sections, _ = recursive_parse(1, toc, 0, total_pages)
    return top_level_sections


def reset_section_levels(
    sections: list[Section], parent_level: str = ""
) -> list[Section]:
    """Recursively reset section levels.

    Args:
        sections: List of sections to reset levels for
        parent_level: Parent level prefix for nested sections

    Returns:
        List of sections with reset level numbering
    """
    sanitized = []
    for i, section in enumerate(sections, 1):
        if getattr(section, "type", None) == "part":
            # Use A, B, C for parts
            current_level = chr(64 + i)  # 65 is ASCII for 'A'
        else:
            current_level = f"{parent_level}{i}" if parent_level else str(i)

        sanitized_subsections = None
        if section.subsections:
            sanitized_subsections = reset_section_levels(
                section.subsections, f"{current_level}."
            )

        sanitized.append(
            Section(
                level=current_level,
                title=section.title,
                start_page=section.start_page,
                end_page=section.end_page,
                subsections=sanitized_subsections,
                type=getattr(section, "type", None),
            )
        )
    return sanitized


# -----------------------------------------------------------------------------
# PUBLIC FUNCTIONS
# -----------------------------------------------------------------------------


def extract_toc(pdf_path: str) -> TOC:
    with fitz.open(pdf_path) as doc:
        total_pages = len(doc)
        toc = doc.get_toc()

    if not toc:
        logger.info("no_toc_found. Attempting AI extraction", pdf_path=pdf_path)
        toc = extract_toc_ai(pdf_path)

    sections = _get_section_hierarchy(toc, total_pages)

    return TOC.model_validate({"sections": sections})


def sanitize_toc(toc: TOC, type: Literal["chapter", "toc", "all"] | None = None) -> TOC:
    """Sanitize the TOC by removing unnecessary sections and subsections."""
    # Extract only the main chapters
    if type is None:
        return toc

    sanitized_sections = extract_chapters(toc.sections)
    if not sanitized_sections:
        raise ValueError("No chapters found in the TOC.")

    sanitized_sections = reset_section_levels(sanitized_sections)
    return TOC.model_validate({"sections": sanitized_sections})
