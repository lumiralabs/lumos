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
from typing import Literal
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
    
    def flatten_sections(self, only_leaf: bool = False):
        return get_sections_flat(self, only_leaf=only_leaf)
    
    def flatten_elements(self):
        elements = []
        for section in self.sections:
            elements.extend(flatten_section_elements(section))
        return elements

    def flatten_chunks(self, dict=True):
        chunks = []
        for section in self.sections:
            chunks.extend(flatten_section_chunks(section))
        
        if dict:
            return [chunk.to_dict() for chunk in chunks]
        return chunks

    def toc(self, level: int | None = None, chapters: bool = False):
        return view_toc(self.metadata.path, level=level, chapters=chapters)

    def to_dict(self):
        book_dict = self.model_dump()   
        recur_to_dict(book_dict)
        
        return book_dict
    

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

# @profile
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
    chapter_pattern = re.compile(r"^chapter \d+(?!.*appendix).*$", re.IGNORECASE)
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
    # Define colors for different levels
    level_colors = ["green", "yellow", "white", "cyan", "red"]
    color = level_colors[(current_level - 1) % len(level_colors)]
    
    for i, section in enumerate(sections, 1):
        if level is None or current_level <= level:
            if parent_tree.label.startswith("("):
                # Get parent number and append current index
                parent_number = parent_tree.label.split(" ")[0].strip("()")
                section_number = f"({parent_number}.{i})"
            else:
                # Top level section
                section_number = f"({i})"
                
            node = parent_tree.add(
                f"[{color}]{section_number} {section.title}[/{color}] [dim italic](Pages: {section.start_page}-{section.end_page})"
            )
            if section.subsections:
                print_section_tree(section.subsections, node, level, current_level + 1)

def print_book_structure(book: Book) -> None:
    metadata = book.metadata
    sections = book.sections
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


def get_sections_flat(book: Book, only_leaf: bool = False) -> list[dict]:
    sections = []
    
    def flatten_section(section: Section, prefix: str = "", number: str = "") -> dict:
        content = ""
        if section.elements:
            content = "\n\n".join(element.text for element in section.elements)
        
        section_dict = {
            "title": section.title,
            "level": number if number else "",
            "start_page": section.start_page,
            "end_page": section.end_page,
            "content": content
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

def view_toc(pdf_path: str, level: int | None = None, chapters: bool = False) -> None:
    sections = get_section_hierarchy(pdf_path)
    chapters_list = extract_chapters(sections)

    console = Console()
    if chapters:
        if chapters_list:
            console.print("[bold green]Chapters Found:[/bold green]")
            for chapter in chapters_list:
                console.print(
                    f"- {chapter.title} (Pages: {chapter.start_page}-{chapter.end_page})"
                )
        else:
            console.print("[bold red]No chapters found.[/bold red]")
    else:
        tree = Tree("[bold magenta]Table of Contents[/bold magenta]")
        print_section_tree(sections, tree, level=level)
        console.print(tree)

# @profile
def parse(pdf_path: str, dev: Literal["partitions", "sections", "chunks", "lessons"] | None = None) -> list[dict]:
    """
    Returns a list of all the chunks in the book.
    """
    metadata = extract_pdf_metadata(pdf_path)
    sections = get_section_hierarchy(pdf_path)
    chapters = extract_chapters(sections)
    
    book_elements = partition(
        filename=pdf_path,
        api_key=os.environ.get("UNSTRUCTURED_API_KEY"),
        partition_endpoint=os.environ.get("UNSTRUCTURED_API_URL"),
        partition_by_api=True,
        strategy="fast",
        include_metadata=True,
    )
    
    print(f"Finished partitioning book elements: {len(book_elements)=}")

    # Clean elements
    book_elements = [
        element
        for element in book_elements
        if element.to_dict()["type"] not in ["Footer", "PageBreak"]
    ]

    # Partition recursively into subsections
    new_chapters = []
    for chapter in chapters:
        chapter.elements = get_elements_for_chapter(book_elements, chapter)
        new_chapter = partition_elements(chapter)
        add_chunks(new_chapter)
        new_chapters.append(new_chapter)

    book = Book(metadata=metadata, sections=new_chapters)
    chunks = book.flatten_chunks(dict=True)
    sections = book.flatten_sections(only_leaf=True)
    
    if dev == "partitions":
        rich_view_chunks(book.flatten_elements())
        return
    elif dev == "sections":
        rich_view_sections(sections)
        return
    elif dev == "lessons":
        view_ai_summaries(sections)
        return
    elif dev == "chunks":
        console = Console()
        console.print()
        for i, chunk in enumerate(chunks):
            console.print(Panel(
                f"[bold cyan]Chunk {i}[/bold cyan]\n\n"
                f"[yellow]Page {chunk['metadata']['page_number']}[/yellow]\n\n"
                f"{chunk['text']}",
                expand=True
            ))
        return
        
    return sections, chunks

async def gather_tasks(leaf_sections) -> list['LessonContent']:
    sem = asyncio.Semaphore(50)
    
    async def bounded_get_lesson(title: str, content: str) -> 'LessonContent':
        async with sem:
            return await get_lesson_content(title, content)
    
    tasks = [bounded_get_lesson(title, content) for title, content in leaf_sections]
    return await asyncio.gather(*tasks)


# async def gather_tasks(leaf_sections) -> list['LessonContent']:
#     tasks = [get_lesson_content(title, content) for title, content in leaf_sections]
#     return await asyncio.gather(*tasks)

# @profile
def view_ai_summaries(sections: list[Section]) -> None:
    loop = asyncio.get_event_loop()
    results = loop.run_until_complete(gather_tasks(sections))
    console = Console()
    for (title, _), lesson in zip(sections, results):
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
            {"role": "system", "content": "You are a helpful assistant and tutot that will help me creating insightful lessons and summaries for technical books. You will be provided with a section of a book and you will need to generate a summary of the content. Be concise and to the point."},
            {"role": "user", "content": input_str},
        ],
        model="gpt-4o-mini",
        response_format=LessonContent,
    )
    return ret


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
    """
    Add chunks recursively to a section and its subsections
    """
    if section.elements:
        section.chunks = chunk_elements(section.elements)
    if section.subsections:
        for subsection in section.subsections:
            add_chunks(subsection)


def rich_view_chunks(chunks: list[dict | Any]) -> None:
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

def rich_view_sections(sections: list[dict]) -> None:
    console = Console()
    table = Table(title="Document Sections", padding=1)
    table.add_column("ID", style="white")
    table.add_column("Level", style="white") 
    table.add_column("Title", style="yellow")
    table.add_column("Content", style="green", no_wrap=False)

    for i, section in enumerate(sections, 1):
        table.add_row(
            str(i),
            section["level"],
            section["title"],
            section["content"][:200] + "..." if len(section["content"]) > 200 else section["content"]
        )

    console.print(table)

def main():
    fire.Fire(
        {
            "toc": view_toc,
            "parse": parse,
        }
    )


if __name__ == "__main__":
    main()

# Usage:
# python -m lumos.book_parser toc .dev/data/asyncio/asyncio.pdf 2 --chapters
# python -m lumos.book_parser parse .dev/data/asyncio/asyncio.pdf --dev=lessons
# python -m lumos.book_parser parse .dev/data/asyncio/asyncio.pdf --dev=chunks
# python -m lumos.book_parser parse .dev/data/asyncio/asyncio.pdf --dev=partitions
