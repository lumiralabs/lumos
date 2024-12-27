from lumos.book.toc import extract_toc, sanitize_toc
from rich.tree import Tree
from rich.console import Console
from lumos.book.visualizer import _build_section_tree


def test_extract_toc():
    with open("tests/tests_book/asyncio_toc.txt", "r") as f:
        asyncio_toc = f.read()

    toc = extract_toc("tests/data/asyncio.pdf")
    tree = Tree("Table of Contents")
    _build_section_tree(toc.sections, tree)
    console = Console(record=True, width=500)
    console.print(tree)
    rich_tree_str = console.export_text()

    assert rich_tree_str == asyncio_toc


def test_sanitize_toc():
    with open("tests/tests_book/asyncio_toc_sanitized.txt", "r") as f:
        expected_toc = f.read()

    _toc = extract_toc("tests/data/asyncio.pdf")
    sanitized_toc = sanitize_toc(_toc)
    tree = Tree("Sanitized Table of Contents")
    _build_section_tree(sanitized_toc.sections, tree)
    console = Console(record=True, width=500)
    console.print(tree)
    rich_tree_str = console.export_text()

    assert rich_tree_str == expected_toc
