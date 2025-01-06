from pydantic import BaseModel, Field, field_validator
from typing import Any, Literal
import structlog

logger = structlog.get_logger(__name__)


class Section(BaseModel):
    level: str = Field(
        description="The level of the section in the TOC like '1.1' or '1.2.1'"
    )
    title: str
    type: Literal["part", "chapter", "section"] | None = Field(
        None,
        description="The type of the section like introduction, contents, chapter, index, appendix, etc.",
    )
    start_page: int | None = None
    end_page: int | None = None
    subsections: list["Section"] | None = None
    chunks: list[Any] | None = None
    elements: list[Any] | None = None

    def flatten_elements(self):
        elements = []
        if self.elements:
            elements.extend(self.elements)
        if self.subsections:
            for subsection in self.subsections:
                elements.extend(subsection.flatten_elements())
        return elements

    def flatten_chunks(self):
        chunks = []
        if self.chunks:
            chunks.extend(
                [
                    chunk.to_dict() if not isinstance(chunk, dict) else chunk
                    for chunk in self.chunks
                ]
            )
        if self.subsections:
            for subsection in self.subsections:
                chunks.extend(subsection.flatten_chunks())
        return chunks

    @field_validator("end_page")
    def end_page_gt_start_page(cls, v, info):
        if v is not None and info.data["start_page"] is not None:
            if v < info.data["start_page"]:
                logger.error(
                    f"End page {v} is less than or equal to start page {info.data['start_page']}"
                )
                return v
                # raise ValueError("End page must be greater than start page")
        return v


class PDFMetadata(BaseModel):
    title: str
    author: str
    subject: str | None
    path: str
    keywords: list[str] | None
    toc: list[list] | None


class TOC(BaseModel):
    sections: list["Section"]

    def from_list(cls, lst: list[list]):
        return cls(sections=lst)


class TOC_LIST(BaseModel):
    sections: list[list]


class Book(BaseModel):
    metadata: PDFMetadata
    sections: list[Section]

    def flatten_sections(self, only_leaf: bool = False):
        return _get_sections_flat(self, only_leaf=only_leaf)

    def flatten_elements(self):
        elements = []
        for section in self.sections:
            elements.extend(section.flatten_elements())
        return elements

    def flatten_chunks(self, dict=True):
        chunks = []
        for section in self.sections:
            chunks.extend(section.flatten_chunks())

        if dict:
            return chunks  # chunks are already converted to dict by Section.flatten_chunks()
        return [
            chunk if isinstance(chunk, dict) else chunk.to_dict() for chunk in chunks
        ]

    def toc(
        self, level: int | None = None, type: Literal["chapter", "toc", "all"] = "all"
    ):
        raise NotImplementedError("TOC extraction is not implemented yet.")

    def to_dict(self):
        book_dict = self.model_dump()

        def _list_2_dict(lst):
            return [item.to_dict() for item in lst]

        def _recur_to_dict(obj):
            if elements := obj.get("elements"):
                obj["elements"] = _list_2_dict(elements)
            if chunks := obj.get("chunks"):
                obj["chunks"] = _list_2_dict(chunks)
            if subsections := obj.get("subsections"):
                for subsection in subsections:
                    _recur_to_dict(subsection)

            if sections := obj.get("sections"):
                for section in sections:
                    _recur_to_dict(section)

        _recur_to_dict(book_dict)

        return book_dict


def _get_sections_flat(book: Book, only_leaf: bool = False) -> list[dict]:
    sections = []

    def flatten_section(section: Section, prefix: str = "", number: str = "") -> dict:
        content = ""
        if section.elements:
            if isinstance(section.elements[0], str):
                content = "\n\n".join(section.elements)
            else:
                content = "\n\n".join(element.text for element in section.elements)

        section_dict = {
            "level": number if number else "",
            "title": section.title,
            "content": content,
            "start_page": section.start_page,
            "end_page": section.end_page,
        }

        if only_leaf:
            if not section.subsections:
                sections.append(section_dict)
        else:
            sections.append(section_dict)

        if section.subsections:
            for i, subsection in enumerate(section.subsections, 1):
                new_number = f"{number}.{i}" if number else str(i)
                flatten_section(subsection, prefix=section.title, number=new_number)

    for i, section in enumerate(book.sections, 1):
        flatten_section(section, number=str(i))

    return sections
