from lumos.book.toc import extract_toc
from rich.tree import Tree
from rich.console import Console
from lumos.book.visualizer import _build_section_tree

# When running pytest from root, relative paths are relative to root directory
with open("tests/tests_book/asyncio_toc.txt", "r") as f:
    asyncio_toc = f.read()


def test_toc_gen():
    toc = extract_toc("tests/data/asyncio.pdf")
    tree = Tree("Table of Contents")
    _build_section_tree(toc.sections, tree)
    console = Console(record=True, width=500)
    console.print(tree)
    rich_tree_str = console.export_text()

    assert rich_tree_str == asyncio_toc
