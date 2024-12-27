import re
import fitz
from pydantic import BaseModel, Field
from rich.tree import Tree
from .models import Section
from lumos import lumos
from .utils import extract_pdf_metadata


def extract_chapters_by_pattern(sections: list[Section]) -> list[Section]:
    """Extract chapters from sections using regex pattern matching."""
    chapter_pattern = re.compile(r"^chapter \d+(?!.*appendix).*$", re.IGNORECASE)
    return [s for s in sections if chapter_pattern.match(s.title)]


def _toc_to_str(sections: list[Section]) -> str:
    """Convert sections to a string representation for AI processing."""
    return "".join(
        [
            f"({idx}) {section.title} (Pages: {section.start_page}-{section.end_page})\n"
            for idx, section in enumerate(sections)
        ]
    )


def extract_chapters_ai(sections: list[Section]) -> list[Section]:
    """Use AI to identify main chapters when pattern matching fails."""
    toc_str = _toc_to_str(sections)

    class BookChapters(BaseModel):
        chapters: list[int] = Field(..., description="List of chapter numbers")

    ret = lumos.call_ai(
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that can identify chapter numbers from a table of contents. Given this table of contents, identify the line numbers (in parentheses) that contain actual numbered chapters (e.g. '01.', '02.' etc). Ignore sections like 'Table of Contents', 'Index', 'Acknowledgements', Appendices, etc. We want the most important chapters relevant for study. Return only a list of integers.",
            },
            {"role": "user", "content": toc_str},
        ],
        model="gpt-4o-mini",
        response_format=BookChapters,
    )

    return [sections[i] for i in ret.chapters]


def extract_chapters(sections: list[Section]) -> list[Section]:
    """Extract main chapters from sections using pattern matching first, then AI if needed."""
    chapters = extract_chapters_by_pattern(sections)
    if not chapters:
        chapters = extract_chapters_ai(sections)
    return chapters


def _build_section_tree(
    sections: list[Section],
    parent_tree: Tree,
    level: int | None = None,
    current_level: int = 1,
) -> None:
    """Build a rich tree visualization of sections."""
    level_colors = ["green", "yellow", "white", "cyan", "red"]
    color = level_colors[(current_level - 1) % len(level_colors)]

    for i, section in enumerate(sections, 1):
        if level is None or current_level <= level:
            if parent_tree.label.startswith("("):
                parent_number = parent_tree.label.split(" ")[0].strip("()")
                section_number = f"({parent_number}.{i})"
            else:
                section_number = f"({i})"

            node = parent_tree.add(
                f"[{color}]{section_number} {section.title}[/{color}] [dim italic](Pages: {section.start_page}-{section.end_page})"
            )
            if section.subsections:
                _build_section_tree(section.subsections, node, level, current_level + 1)


class TOCExtractor:
    """Class to handle PDF table of contents extraction and visualization."""

    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.metadata = extract_pdf_metadata(pdf_path)
        with fitz.open(pdf_path) as doc:
            self.total_pages = len(doc)
            self.toc = doc.get_toc()

    def get_section_hierarchy(self) -> list[Section]:
        """Get the complete section hierarchy from TOC."""
        if not self.toc:
            return []
        return self._build_section_hierarchy(self.toc, self.total_pages)

    def _build_section_hierarchy(
        self, toc: list[list], total_pages: int
    ) -> list[Section]:
        """Build a hierarchical structure of sections from the TOC."""

        def recursive_parse(level, toc, index, parent_end_page):
            sections = []
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

                subsection, next_index = recursive_parse(
                    curr_level + 1, toc, index + 1, end_page
                )

                sections.append(
                    Section(
                        title=title,
                        start_page=page,
                        end_page=end_page,
                        subsections=subsection or None,
                    )
                )
                index = next_index
            return sections, index

        top_level_sections, _ = recursive_parse(1, toc, 0, total_pages)
        return top_level_sections

    def get_chapters(self) -> list[Section]:
        """Extract main chapters from the TOC."""
        sections = self.get_section_hierarchy()
        return extract_chapters(sections)
