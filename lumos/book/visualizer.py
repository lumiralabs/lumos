from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.tree import Tree
from typing import Any
from .models import Book, Section


def rich_view_chunks(chunks: list[dict | Any]) -> None:
    """Display chunks in a rich table format."""
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
        table.add_row(
            str(i),
            _chunk.get("type", "<no type>"),
            _chunk.get("text", "<no text>"),
            str(page_number),
        )

    console.print(table)


def rich_view_sections(sections: list[dict]) -> None:
    """Display sections in a rich table format."""
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
            section["content"][:200] + "..."
            if len(section["content"]) > 200
            else section["content"],
        )

    console.print(table)


def rich_view_toc_sections(
    sections: list[Section],
    level: int | None = None,
) -> None:
    assert isinstance(sections, list)
    console = Console()

    tree = Tree("[bold magenta]Table of Contents[/bold magenta]")
    _build_section_tree(sections, tree, level=level)

    console.print(tree)
    return tree


def _build_section_tree(
    sections: list[Section],
    parent_tree: Tree,
    level: int | None = None,
    current_level: int = 1,
) -> None:
    """Build a rich tree visualization of sections.

    Args:
        sections: List of sections to display
        parent_tree: Parent tree node to add sections to
        level: Maximum depth level to display (None for all levels)
        current_level: Current depth level being processed
    """
    level_colors = ["green", "yellow", "white", "cyan", "red"]
    color = level_colors[(current_level - 1) % len(level_colors)]

    for section in sections:
        if level is None or current_level <= level:
            node = parent_tree.add(
                f"[{color}]({section.level}) {section.title}[/{color}] [dim italic](Pages: {section.start_page}-{section.end_page})"
            )
            if section.subsections:
                _build_section_tree(section.subsections, node, level, current_level + 1)


def print_book_structure(book: Book) -> None:
    """Print the complete book structure including metadata and sections."""
    metadata = book.metadata
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

    # Print sections
    rich_view_sections(
        [
            {
                "level": "",
                "title": section.title,
                "content": "\n".join(e.text for e in section.elements)
                if section.elements
                else "",
            }
            for section in book.sections
        ]
    )
