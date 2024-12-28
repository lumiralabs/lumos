import pytest
from rich.tree import Tree
from rich.console import Console
from lumos.book.toc_ai import extract_toc, detect_toc_pages
from lumos.book.visualizer import _build_section_tree


@pytest.mark.parametrize(
    "book_name,expected_pages", [("asyncio", [7, 8]), ("almanack", [6, 7])]
)
def test_detect_toc_pages(book_name, expected_pages):
    pdf_path = f"tests/data/{book_name}.pdf"
    toc_pages = detect_toc_pages(pdf_path)
    assert toc_pages == expected_pages


@pytest.mark.parametrize(
    "book_name,page_ranges", [("asyncio", [7, 8]), ("almanack", [6, 7])]
)
def test_extract_toc(book_name, page_ranges):
    toc_file = f"tests/data/{book_name}_toc.txt"
    with open(toc_file, "r") as f:
        expected_toc = f.read()

    pdf_path = f"tests/data/{book_name}.pdf"
    toc = extract_toc(pdf_path, page_ranges)

    tree = Tree("Table of Contents")
    _build_section_tree(toc.sections, tree)

    console = Console(record=True, width=500)
    console.print(tree)
    rich_tree_str = console.export_text()

    with open(f"tests/data/{book_name}_toc_out.txt", "w") as f:
        f.write(rich_tree_str)

    assert rich_tree_str.strip() == expected_toc.strip()
