import pytest
from lumos.book.toc import extract_toc_from_metadata, sanitize_toc
from rich.tree import Tree
from rich.console import Console
from lumos.book.visualizer import _build_section_tree


@pytest.mark.parametrize("book_name", ["asyncio", "almanack"])
def test_extract_toc(book_name):
    toc_file = f"tests/data/{book_name}_toc.txt"
    with open(toc_file, "r") as f:
        expected_toc = f.read()

    pdf_path = f"tests/data/{book_name}.pdf"
    toc = extract_toc_from_metadata(pdf_path)
    tree = Tree("Table of Contents")
    _build_section_tree(toc.sections, tree)
    console = Console(record=True, width=500)
    console.print(tree)
    rich_tree_str = console.export_text()

    assert rich_tree_str.strip() == expected_toc.strip()


@pytest.mark.parametrize("book_name", ["asyncio", "almanack"])
def test_sanitize_toc(book_name):
    expected_file = f"tests/data/{book_name}_toc_sanitized.txt"
    with open(expected_file, "r") as f:
        expected_toc = f.read()

    pdf_path = f"tests/data/{book_name}.pdf"
    toc = extract_toc_from_metadata(pdf_path)
    sanitized_toc = sanitize_toc(toc, type="chapter")
    tree = Tree("Sanitized Table of Contents")
    _build_section_tree(sanitized_toc.sections, tree)
    console = Console(record=True, width=500)
    console.print(tree)
    rich_tree_str = console.export_text()

    output_file = f"tests/data/{book_name}_toc_sanitized_out.txt"
    with open(output_file, "w") as f:
        f.write(rich_tree_str)

    assert rich_tree_str.strip() == expected_toc.strip()
