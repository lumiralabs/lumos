import pytest
from lumos.book.toc_ai import detect_toc_pages, get_offset, extract_toc_ai
import fitz
from rich.tree import Tree
from rich.console import Console

from lumos.book.toc import sanitize_toc, _get_section_hierarchy, TOC
from lumos.book.visualizer import _build_section_tree


@pytest.mark.parametrize(
    "book_name,expected_pages",
    [("asyncio", [7, 8]), ("almanack", [6, 7]), ("portfolio", [9, 10, 11, 12])],
)
def test_detect_toc_pages(book_name, expected_pages):
    pdf_path = f"tests/data/{book_name}.pdf"
    toc_pages = detect_toc_pages(pdf_path)
    assert toc_pages == expected_pages


@pytest.mark.parametrize(
    "book_name, range, expected_offset",
    [("asyncio", [7, 8], 12), ("almanack", [6, 7], 0), ("portfolio", [9, 12], 12)],
)
def test_get_offset(book_name, range, expected_offset):
    pdf_path = f"tests/data/{book_name}.pdf"
    with fitz.open(pdf_path) as doc:
        pages_str = [p.get_text() for p in doc.pages()]

    toc_list = extract_toc_ai(pdf_path, range)
    offset = get_offset(toc_list, pages_str, start_offset=max(range))
    assert offset == expected_offset


@pytest.mark.parametrize(
    "book_name,page_ranges",
    [
        # ("asyncio", [7, 8]),
        # ("almanack", [6, 7]),
        ("portfolio", [9, 12])
    ],
)
def test_sanitize_toc(book_name, page_ranges):
    toc_file = f"tests/data/{book_name}_toc_sanitized.txt"
    with open(toc_file, "r") as f:
        expected_toc = f.read()

    pdf_path = f"tests/data/{book_name}.pdf"
    with fitz.open(pdf_path) as doc:
        total_pages = len(doc)
    toc_list = extract_toc_ai(pdf_path, page_ranges)
    sections = _get_section_hierarchy(toc_list, total_pages)
    toc = TOC.model_validate({"sections": sections})

    toc = sanitize_toc(toc, type="chapter")

    tree = Tree("Table of Contents")
    _build_section_tree(toc.sections, tree)

    console = Console(record=True, width=500)
    console.print(tree)
    rich_tree_str = console.export_text()

    with open(f"tests/data/{book_name}_toc_ai_out.txt", "w") as f:
        f.write(rich_tree_str)

    assert rich_tree_str.strip() == expected_toc.strip()
