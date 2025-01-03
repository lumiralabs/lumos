import os
import pickle
from typing import Literal
import fire
from unstructured.partition.auto import partition
from .models import Book
from .toc import sanitize_toc, extract_toc
from .toc_ai import extract_toc as extract_toc_ai
from .element_processor import get_elements_for_chapter, partition_elements, add_chunks
from .visualizer import rich_view_chunks, rich_view_sections, rich_view_toc_sections
from .utils import extract_pdf_metadata
import structlog

logger = structlog.get_logger()


def from_pdf_path(pdf_path: str) -> Book:
    """Create a Book object from a PDF file."""
    metadata = extract_pdf_metadata(pdf_path)

    # toc = extract_toc_ai(pdf_path)
    try:
        toc = extract_toc(pdf_path)
    except Exception as e:
        logger.error("error_extracting_toc. Attempting AI extraction", error=e)
        try:
            toc = extract_toc_ai(pdf_path)
        except Exception as e:
            logger.error("error_extracting_toc_ai", error=e)
            raise

    print("TOC:")
    rich_view_toc_sections(toc.sections)
    toc = sanitize_toc(toc, type="chapter")
    print("Sanitized TOC:")
    rich_view_toc_sections(toc.sections)
    chapters = toc.sections

    # Extract and process book elements
    logger.info("[book-parser] Extracting book elements", pdf_path=pdf_path)
    book_elements = partition(
        filename=pdf_path,
        api_key=os.environ.get("UNSTRUCTURED_API_KEY"),
        partition_endpoint=os.environ.get("UNSTRUCTURED_API_URL"),
        partition_by_api=True,
        strategy="fast",
        include_metadata=True,
    )
    logger.info("[book-parser] Extracted book elements", len=len(book_elements))

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

    return Book(metadata=metadata, sections=new_chapters)


def parse(pdf_path: str):
    book = from_pdf_path(pdf_path)
    return book.flatten_sections(only_leaf=True), book.flatten_chunks(dict=True)


def dev(
    pdf_path: str,
    type: Literal["partitions", "sections", "chunks"] | None = None,
) -> None:
    """Development function for testing different aspects of the parser."""
    book_pickle_path = os.path.basename(pdf_path).replace(".pdf", ".pickle")
    if os.path.exists(book_pickle_path):
        print(f"Loading book from {book_pickle_path}...")
        with open(book_pickle_path, "rb") as f:
            book = pickle.load(f)
    else:
        print(f"Parsing book from {pdf_path}...")
        book = from_pdf_path(pdf_path)
        print(f"Dumping book to {book_pickle_path}...")
        with open(book_pickle_path, "wb") as f:
            pickle.dump(book, f)

    chunks = book.flatten_chunks(dict=True)
    sections = book.flatten_sections(only_leaf=True)

    # Print statistics
    from rich.console import Console
    from rich.table import Table

    console = Console()
    table = Table(title="Book Statistics", padding=1)
    table.add_column("Type", style="cyan")
    table.add_column("Count", style="yellow")

    table.add_row("Elements", str(len(book.flatten_elements())))
    table.add_row("Chunks", str(len(chunks)))
    table.add_row("Sections", str(len(sections)))

    console.print(table)
    console.print()

    # Show requested view
    if type == "partitions":
        rich_view_chunks(book.flatten_elements())
    elif type == "sections":
        rich_view_sections(sections)
    elif type == "chunks":
        rich_view_chunks(chunks)


if __name__ == "__main__":
    fire.Fire(dev)
