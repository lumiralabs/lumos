import os
import re
import fitz
from pydantic import BaseModel
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.tree import Tree
import fire
import json
import fitz
from typing import Any
import copy
from unstructured.partition.auto import partition
from unstructured.chunking.title import chunk_by_title
from lumos import lumos
from pydantic import BaseModel, Field
import asyncio

class Section(BaseModel):
    title: str
    start_page: int
    end_page: int
    subsections: list["Section"] | None = None
    chunks: list[Any] | None = None
    elements: list[Any] | None = None


class PDFMetadata(BaseModel):
    title: str
    author: str
    subject: str | None
    path: str
    keywords: list[str] | None
    toc: list[list] | None


class Book(BaseModel):
    metadata: PDFMetadata
    sections: list[Section]

    def flatten_elements(self):
        elements = []
        for section in self.sections:
            elements.extend(flatten_section_elements(section))
        return elements

    def flatten_chunks(self):
        chunks = []
        for section in self.sections:
            chunks.extend(flatten_section_chunks(section))
        return chunks

    def toc(self, level: int | None = None):
        return view_toc(self.metadata.path, level=level)

    def to_json(self):
        book_dict = self.model_dump()   
        recur_to_dict(book_dict)
        
        return json.dumps(book_dict, indent=2)

def list_2_dict(lst):
    return [item.to_dict() for item in lst]

def recur_to_dict(obj):
    if elements := obj.get("elements"):
        obj["elements"] = list_2_dict(elements)
    if chunks := obj.get("chunks"):
        obj["chunks"] = list_2_dict(chunks)
    if subsections := obj.get("subsections"):
        for subsection in subsections:
            recur_to_dict(subsection)
            
    if sections := obj.get("sections"):
        for section in sections:
            recur_to_dict(section)


def flatten_section_chunks(section: Section) -> list[Any]:
    chunks = []
    if section.chunks:
        chunks.extend(section.chunks)
    if section.subsections:
        for subsection in section.subsections:
            chunks.extend(flatten_section_chunks(subsection))
    return chunks


def flatten_section_elements(section: Section) -> list[Any]:
    elements = []
    if section.elements:
        elements.extend(section.elements)
    if section.subsections:
        for subsection in section.subsections:
            elements.extend(flatten_section_elements(subsection))
    return elements


def extract_pdf_metadata(pdf_path: str) -> PDFMetadata:
    with fitz.open(pdf_path) as doc:
        raw_title = doc.metadata.get("title", "")
        raw_author = doc.metadata.get("author", "")
        raw_subject = doc.metadata.get("subject")
        raw_keywords = doc.metadata.get("keywords", "")
        keywords = raw_keywords.split(",") if raw_keywords else None

        return PDFMetadata(
            title=raw_title,
            author=raw_author,
            subject=raw_subject,
            keywords=keywords,
            path=pdf_path,
            toc=doc.get_toc(),
        )


def get_leaf_sections(section) -> list[tuple[str, list]]:
    """Get all leaf sections (those without subsections) and their elements."""
    results = []
    
    if section.get('elements') and not section.get('subsections'):
        # This is a leaf section - collect title and elements
        ele_str = "\n\n".join([element['text'] for element in section['elements']])
        results.append((section['title'], ele_str))
    
    # Recursively process subsections
    if section.get('subsections'):
        for subsection in section['subsections']:
            results.extend(get_leaf_sections(subsection))
            
    return results


def build_section_hierarchy(toc: list[list], total_pages: int) -> list[Section]:
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


def get_section_hierarchy(pdf_path: str) -> list[Section]:
    metadata = extract_pdf_metadata(pdf_path)
    if not metadata.toc:
        return []

    with fitz.open(pdf_path) as doc:
        total_pages = len(doc)
    return build_section_hierarchy(metadata.toc, total_pages)


def extract_toc(sections: list[Section]) -> Section:
    toc_pattern = re.compile(r"^table of contents$", re.IGNORECASE)
    toc_sections = []

    for section in sections:
        if toc_pattern.match(section.title):
            toc_sections.append(section)
    assert len(toc_sections) == 1, "Expected exactly one TOC section"
    return toc_sections[0]


def extract_chapters(sections: list[Section]) -> list[Section]:
    chapter_pattern = re.compile(r"^chapter \d+", re.IGNORECASE)
    chapters = []

    for section in sections:
        if chapter_pattern.match(section.title):
            chapters.append(section)
    return chapters


def extract_section_text(pdf_path: str, section: Section) -> str:
    """
    start and end are 1-indexed and inclusive
    """
    texts = []
    with fitz.open(pdf_path) as doc:
        for page_num in range(section.start_page - 1, section.end_page):
            texts.append(doc.load_page(page_num).get_text())
    return "\n".join(texts)


def extract_section_text_from_middle(pdf_path: str, section: Section) -> str:
    """
    start and end are 1-indexed and inclusive
    Extracts the text from the middle of a page by detecting the start and end of the section
    """
    texts = []
    with fitz.open(pdf_path) as doc:
        for page_num in range(section.start_page - 1, section.end_page):
            texts.append(doc.load_page(page_num).get_text())

    first_page = texts[0]
    first_page_start = first_page.find(section.title)
    cropped_first_page = first_page[first_page_start:]
    texts[0] = cropped_first_page

    return "\n".join(texts)


def print_section_tree(
    sections: list[Section], parent_tree=None, level: int | None = None, current_level=1
):
    for section in sections:
        if level is None or current_level <= level:
            node = parent_tree.add(
                f"{section.title} [dim](Pages: {section.start_page}-{section.end_page})"
            )
            if section.subsections:
                print_section_tree(section.subsections, node, level, current_level + 1)


def print_book_structure(metadata: PDFMetadata, sections: list[Section]) -> None:
    console = Console()

    # Metadata Panel
    metadata_content = (
        f"[bold blue]Title:[/bold blue] {metadata.title}\n"
        f"[bold blue]Author:[/bold blue] {metadata.author}\n"
        f"[bold blue]Subject:[/bold blue] {metadata.subject}"
    )
    console.print(
        Panel(metadata_content, title="Document Metadata", border_style="blue")
    )
    console.print()

    # Section Tree
    tree = Tree("[bold magenta]Document Structure[/bold magenta]")
    print_section_tree(sections, tree)
    console.print(tree)


def view_toc(pdf_path: str, level: int | None = None) -> None:
    metadata = extract_pdf_metadata(pdf_path)
    sections = get_section_hierarchy(pdf_path)

    console = Console()
    tree = Tree("[bold magenta]Table of Contents[/bold magenta]")
    print_section_tree(sections, tree, level=level)
    console.print(tree)


def list_chapters(pdf_path: str) -> None:
    sections = get_section_hierarchy(pdf_path)
    chapters = extract_chapters(sections)

    console = Console()
    if chapters:
        console.print("[bold green]Chapters Found:[/bold green]")
        for chapter in chapters:
            console.print(
                f"- {chapter.title} (Pages: {chapter.start_page}-{chapter.end_page})"
            )
    else:
        console.print("[bold red]No chapters found.[/bold red]")

@profile
def parse(pdf_path: str) -> None:
    metadata = extract_pdf_metadata(pdf_path)
    sections = get_section_hierarchy(pdf_path)
    chapters = extract_chapters(sections)

    book_elements = partition(
        filename=pdf_path,
        # filename="./.dev/output/asyncio.md",
        api_key=os.environ.get("UNSTRUCTURED_API_KEY"),
        partition_endpoint=os.environ.get("UNSTRUCTURED_API_URL"),
        partition_by_api=True,
        # include_page_breaks=True,
        include_metadata=True,
    )

    # Clean elements
    book_elements = [
        element
        for element in book_elements
        if element.to_dict()["type"] not in ["Footer", "PageBreak"]
    ]

    new_chapters = []
    for chapter in chapters:
        chapter.elements = get_elements_for_chapter(book_elements, chapter)
        new_chapters.append(partition_elements(chapter))

    book = Book(metadata=metadata, sections=new_chapters)

    for section in book.sections:
        add_chunks(section)

    book_dict = book.model_dump()
    recur_to_dict(book_dict)
    
    get_lessons(book_dict)
    

async def gather_tasks(leaf_sections) -> None:
    tasks = [get_lesson_content(title, content) for title, content in leaf_sections]
    return await asyncio.gather(*tasks)

@profile
def get_lessons(book_dict: dict) -> None:
    leaf_sections = []
    for section in book_dict['sections']:
        leaf_sections.extend(get_leaf_sections(section))
    results = asyncio.run(gather_tasks(leaf_sections))
    console = Console()
    for (title, content), lesson in zip(leaf_sections, results):
        console.print()
        console.print(Panel(
            f"[bold magenta]{title}[/bold magenta]\n\n"
        f"[yellow]Description:[/yellow] {lesson.description}\n\n"
        f"[green]Summary:[/green] {lesson.summary}",
        expand=True
    ))

class LessonContent(BaseModel):
    description: str = Field(..., description="One or two line description of the content. Get the most information across.")
    summary: str = Field(..., description="A concise summary of the content. Be to the point and concise.")

async def get_lesson_content(title, content):
    input_str = """Generate a summary of the content: 
    <Title>
    {title}
    </Title>

    <Content>
    {content}
    </Content>""".format(title=title, content=content)

    ret = await lumos.call_ai_async(
        messages=[
            {"role": "system", "content": "You are a helpful assistant that will help me creating insightful lessons and summaries for technical books. You will be provided with a section of a book and you will need to generate a summary of the content. Be concise and to the point. "},
            {"role": "user", "content": input_str},
        ],
        model="gpt-4o",
        response_format=LessonContent,
    )
    return ret

def trim_pdf_to_sections(
    input_pdf_path: str, output_pdf_path: str, sections: list[Section]
) -> None:
    """
    Create a new PDF containing only the pages from the specified sections and save section metadata.

    Args:
        input_pdf_path: Path to source PDF file
        output_pdf_path: Path where trimmed PDF will be saved
        sections: List of sections to keep
    """
    # Open source PDF
    doc = fitz.open(input_pdf_path)

    # Create new PDF for output
    out_doc = fitz.open()

    # Copy metadata from source to output
    for key, value in doc.metadata.items():
        out_doc.set_metadata({key: value})

    # Get all page ranges from sections and create page mapping
    page_ranges = []
    old_to_new_page = {}  # Maps original page numbers to new ones
    new_page_count = 0

    # Keep track of updated section metadata
    updated_sections = []

    for section in sections:
        # Convert from 1-based to 0-based page numbers
        start_idx = section.start_page - 1
        end_idx = section.end_page - 1
        page_ranges.append((start_idx, end_idx))

        # Calculate new page numbers for this section
        new_start_page = new_page_count + 1  # Convert to 1-based

        # Build page number mapping
        for old_page in range(start_idx, end_idx + 1):
            old_to_new_page[old_page] = new_page_count
            new_page_count += 1

        new_end_page = new_page_count  # Already 1-based since we incremented

        # Create updated section with new page numbers
        updated_section = Section(
            title=section.title,
            start_page=new_start_page,
            end_page=new_end_page,
            subsections=section.subsections,
        )
        updated_sections.append(updated_section)

    # Add pages that fall within any range
    for page_num in range(doc.page_count):
        for start_idx, end_idx in page_ranges:
            if start_idx <= page_num <= end_idx:
                out_doc.insert_pdf(doc, from_page=page_num, to_page=page_num)
                break

    # Update and copy table of contents with new page numbers
    toc = doc.get_toc()
    if toc:
        new_toc = []
        for level, title, page in toc:
            # Convert to 0-based for lookup
            zero_based_page = page - 1
            if zero_based_page in old_to_new_page:
                # Convert back to 1-based for TOC
                new_page = old_to_new_page[zero_based_page] + 1
                new_toc.append([level, title, new_page])

        if new_toc:
            out_doc.set_toc(new_toc)

    # Save the trimmed PDF
    out_doc.save(output_pdf_path)

    # Save section metadata
    metadata_path = output_pdf_path.rsplit(".", 1)[0] + "_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(
            {
                "sections": [section.dict() for section in updated_sections],
                "original_pdf": input_pdf_path,
                "page_mapping": {
                    str(k): v for k, v in old_to_new_page.items()
                },  # Convert keys to strings for JSON
            },
            f,
            indent=2,
        )

    # Close both documents
    doc.close()
    out_doc.close()

    return updated_sections


def get_chunks_for_section(chunks, section) -> list[dict]:
    """
    Get unstructured chunks for a given section
    """
    ret = []
    found_title = False
    print(f"Looking for section: {section.title}")
    for chunk in chunks:
        page_number = chunk["metadata"]["page_number"]

        # For first page, only include chunks after section title
        if page_number == section.start_page:
            if chunk["text"] == section.title:
                print(f"Found title on page {page_number}")
                found_title = True
            if not found_title:
                continue

        # For other pages, include all non-footer chunks up to end page
        if (
            chunk["type"] != "Footer"
            and page_number >= section.start_page
            and page_number <= section.end_page
        ):
            ret.append(chunk)

    return ret


def get_elements_for_chapter(
    elements: list[Any] | list[dict], section: Section
) -> list:
    """
    Chapter sections have clear page boundaries, so we can just filter by page numbers
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

    Args:
        elements: List of document elements (dicts with 'text' and 'metadata')
        section: Section object containing title, start page, and subsections

    Returns:
        Section: New section with elements partitioned through all levels of subsections
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
    return chunk_by_title(
        elements, max_characters=1000, new_after_n_chars=500, multipage_sections=True
    )


def add_chunks(section):
    if section.elements:
        section.chunks = chunk_elements(section.elements)
    if section.subsections:
        for subsection in section.subsections:
            add_chunks(subsection)


def view_chunks(chunks: list[dict | Any]) -> None:
    console = Console()
    table = Table(title="Document Chunks", padding=1)
    table.add_column("#", style="cyan")
    table.add_column("Type", style="cyan")
    table.add_column("Text", style="white", no_wrap=False)
    table.add_column("Page", style="yellow", no_wrap=False)

    for i, chunk in enumerate(chunks, 1):
        _chunk = chunk.to_dict() if not isinstance(chunk, dict) else chunk
        page_number = (
            _chunk["metadata"]["page_number"]
            if "page_number" in _chunk["metadata"]
            else ""
        )
        table.add_row(str(i), _chunk["type"], _chunk["text"], str(page_number))

    console.print(table)


def main():
    fire.Fire(
        {
            "toc": view_toc,
            "parse": parse,
            "chapters": list_chapters,
        }
    )


if __name__ == "__main__":
    main()

# Usage:
# python book_parser.py toc .dev/data/asyncio/asyncio.pdf 2
# python book_parser.py chapters .dev/data/asyncio/asyncio.pdf
