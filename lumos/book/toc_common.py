from .models import Section

def toc_list_to_toc_sections(toc: list[list], total_pages: int | None = None) -> list[Section]:
    """
    Convert a flat TOC list into a structured section hierarchy.
    """
    def recursive_parse(level: int, toc: list[list], index: int, parent_end_page: int | None, parent_level: str = ""):
        sections = []
        section_num = 1

        while index < len(toc):
            curr_level, title, page = toc[index]

            if curr_level < level:
                break

            if total_pages is not None and page is None:
                index += 1
                continue

            current_level = f"{parent_level}{section_num}" if parent_level else str(section_num)

            start_page = page if total_pages is not None else None
            end_page = parent_end_page

            for next_index in range(index + 1, len(toc)):
                next_level, _, next_page = toc[next_index]
                if next_level <= curr_level and next_page is not None:
                    end_page = next_page - 1
                    break

            subsection, next_index = recursive_parse(curr_level + 1, toc, index + 1, end_page, f"{current_level}.")

            sections.append(Section(
                level=current_level,
                title=title,
                start_page=start_page,
                end_page=end_page,
                subsections=subsection or None
            ))

            section_num += 1
            index = next_index

        return sections, index

    top_level_sections, _ = recursive_parse(1, toc, 0, total_pages)
    return top_level_sections
