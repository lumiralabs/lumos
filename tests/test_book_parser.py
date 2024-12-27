from lumos.book.toc import TOCExtractor


def test_toc_gen():
    toc_extractor = TOCExtractor("tests/test_data/asyncio.pdf")
    toc_extractor.get_section_hierarchy()
