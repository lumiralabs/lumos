import os
import pickle
import asyncio
import structlog
from lumos import lumos
from pydantic import BaseModel, Field
from typing import Literal
from rich.table import Table
from rich.console import Console
import fire
from datetime import datetime
import itertools

logger = structlog.get_logger()


# ------------------------
# Pydantic Models
# ------------------------
class Lesson(BaseModel):
    description: str = Field(
        ...,
        description=(
            "One or two line description of the content. Get the most information across."
        ),
    )
    summary: str = Field(
        ...,
        description="A concise summary of the content. Be to the point and concise.",
    )
    types: list[
        Literal[
            "introduction",
            "tutorial",
            "walkthrough",
            "conclusion",
            "example",
            "concept",
            "quiz",
            "story",
            "theory",
            "best_practices",
            "comparison",
            "history",
            "faq",
            "math",
            "code",
            "other",
        ]
    ] = Field(
        ...,
        description=(
            "The type of the section with respect to the book. What is it mainly talking about? "
            "We want to categorize the lessons into types. It can be a analogy of some core concept, "
            "a tutorial, walkthrough, conclusions, etc. It can be a list of types. "
            "Be succinct and insightful."
        ),
    )
    importance_reasoning: str = Field(
        ...,
        description=(
            "Why is the content important? Why is it not important? Be succinct and insightful."
        ),
    )
    important: bool = Field(
        ...,
        description=(
            "Whether the content contains substantial technical material. Content is important if it "
            "explains core concepts, technical implementations, or key design patterns. "
            "Content is not important if it's introductory material, chapter summaries, "
            "or high-level overviews without technical depth."
        ),
    )


# ------------------------
# Async Tasks
# ------------------------
async def section_to_lesson(section: dict) -> Lesson:
    """
    Asynchronously request a summary/analysis from your AI assistant.
    """
    title = section["title"]
    content = section["content"]

    input_str = (
        f"Generate a summary of the content:\n"
        f"<Title>\n{title}\n</Title>\n\n"
        f"<Content>\n{content}\n</Content>"
    )

    return await lumos.call_ai_async(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant and tutor that will help me create "
                    "insightful lessons and summaries for technical books. You will be "
                    "provided with a section of a book and need to generate a few things "
                    "like description, summary, the type of section, and whether it is important. "
                    "We will be creating lessons and quizzes from all the sections and need to "
                    "rank them based on their importance. Be concise and to the point."
                ),
            },
            {"role": "user", "content": input_str},
        ],
        model="gpt-4o",
        response_format=Lesson,
    )


async def sections_to_lessons(sections: list[dict]) -> list[Lesson]:
    """
    Process sections concurrently (up to a limited number of tasks at once).
    """
    semaphore = asyncio.Semaphore(30)

    async def process_section(section: dict) -> Lesson:
        async with semaphore:
            return await section_to_lesson(section)

    tasks = [asyncio.create_task(process_section(section)) for section in sections]
    return await asyncio.gather(*tasks)


# ------------------------
# Rich View
# ------------------------
def rich_view_lessons(sections: list[dict], results: list[Lesson]) -> None:
    """
    Display results as a Rich table in the console.
    """
    table = Table(title="Lessons", padding=1)
    table.add_column("Level", style="white", no_wrap=False)
    table.add_column("Title", style="cyan", no_wrap=False)
    table.add_column("Pages", style="green", no_wrap=False)
    table.add_column("Description", style="magenta", no_wrap=False)
    table.add_column("Types", style="yellow", no_wrap=False)
    table.add_column("Importance Reasoning", style="yellow", no_wrap=False)
    table.add_column("Important", no_wrap=False)

    for section, lesson in zip(sections, results):
        table.add_row(
            section["level"],
            section["title"],
            f"{section['start_page']}-{section['end_page']}",
            lesson.description,
            ", ".join(lesson.types),
            lesson.importance_reasoning,
            "[green]Yes[/green]" if lesson.important else "[red]No[/red]",
        )

    Console().print(table)


def rich_view_sorted_sections(sections: list[dict], scores: dict) -> None:
    """
    Display sorted sections as a Rich table in the console with voting statistics.
    """
    # Create main sections table
    table = Table(title="Sorted Sections by Importance", padding=1)
    table.add_column("Rank", style="white", no_wrap=True)
    table.add_column("Level", style="white", no_wrap=False)
    table.add_column("Title", style="cyan", no_wrap=False)
    table.add_column("Pages", style="green", no_wrap=False)
    table.add_column("Score", style="yellow", no_wrap=False)

    total_score = sum(scores.values())
    for idx, section in enumerate(sections, 1):
        score = scores[idx - 1]  # idx-1 because we're using 1-based ranking
        percentage = (score / total_score) * 100 if total_score > 0 else 0
        table.add_row(
            str(idx),
            section["level"],
            section["title"],
            f"{section['start_page']}-{section['end_page']}",
            f"{score:.1f} ({percentage:.1f}%)",
        )

    # Create statistics table
    stats_table = Table(title="Comparison Statistics", padding=1)
    stats_table.add_column("Metric", style="cyan")
    stats_table.add_column("Value", style="green")

    n = len(sections)
    total_comparisons = (n * (n - 1)) // 2
    avg_score = total_score / n if n > 0 else 0
    max_score = max(scores.values()) if scores else 0
    min_score = min(scores.values()) if scores else 0

    stats_table.add_row("Total Sections", str(n))
    stats_table.add_row("Total Comparisons", str(total_comparisons))
    stats_table.add_row("Average Score", f"{avg_score:.2f}")
    stats_table.add_row("Max Score", f"{max_score:.2f}")
    stats_table.add_row("Min Score", f"{min_score:.2f}")
    stats_table.add_row("Score Range", f"{max_score - min_score:.2f}")

    console = Console()
    console.print(table)
    console.print("\n")
    console.print(stats_table)


# ------------------------
# Sorting Logic
# ------------------------
class ImportanceComparison(BaseModel):
    importance_reasoning: str = Field(
        ...,
        description=(
            "Why is the content important? Why is it not important? Be succinct and insightful."
        ),
    )
    importance_comparison: int = Field(
        ..., description="The comparison result: -1, 0, or 1."
    )


async def pairwise_comparison_ai(section_a: dict, section_b: dict) -> int:
    """
    Asynchronously compare the importance of two sections using AI,
    returning -1, 0, or 1 accordingly.
    """
    prompt = (
        f"<Section A>\n{section_a['content']}\n</Section A>\n\n"
        f"<Section B>\n{section_b['content']}\n</Section B>\n\n"
    )

    response = await lumos.call_ai_async(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant that compares the importance of two sections. "
                    "Whether the content contains substantial technical material. "
                    "Content is important if it explains core concepts, technical implementations, "
                    "or key design patterns. Content is not important if it's introductory material, "
                    "chapter summaries, or high-level overviews without technical depth. "
                    "We will be creating lessons and quizzes from all the sections and need to "
                    "rank them based on their importance. "
                    "Return -1 if Section A is less important, 0 if equally important, "
                    "and 1 if Section A is more important."
                    "Be opinionated and decisive."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        model="gpt-4o",
        response_format=ImportanceComparison,
    )

    cmp_val = response.importance_comparison
    logger.info(
        "ai_comparison_made",
        section_a_title=section_a["title"],
        section_b_title=section_b["title"],
        result=cmp_val,
    )
    return cmp_val


async def sort_sections(sections: list[dict]) -> tuple[list[dict], dict]:
    """Sort sections by sampling n*(n-1)/2 comparisons and aggregating scores"""
    # Generate all pairs of sections
    pairs = list(itertools.combinations(range(len(sections)), 2))

    # Create tasks for all comparisons
    async def compare_pair(i: int, j: int):
        result = await pairwise_comparison_ai(sections[i], sections[j])
        return i, j, result

    tasks = [compare_pair(i, j) for i, j in pairs]
    results = await asyncio.gather(*tasks)

    # Calculate scores for each section
    scores = {i: 0 for i in range(len(sections))}
    for i, j, result in results:
        if result == 1:
            scores[i] += 1
        elif result == -1:
            scores[j] += 1
        else:  # result == 0
            scores[i] += 0.5
            scores[j] += 0.5

    # Create section-score pairs and sort by score
    section_scores = list(zip(sections, scores.values()))
    sorted_section_scores = sorted(section_scores, key=lambda x: x[1], reverse=True)

    # Unzip the sorted sections and scores
    sorted_sections = [section for section, _ in sorted_section_scores]
    sorted_scores = {i: score for i, (_, score) in enumerate(sorted_section_scores)}

    return sorted_sections, sorted_scores


# ------------------------
# MAIN LOGIC
# ------------------------
async def sort_main(pdf_path: str = "dev/data/asyncio/asyncio.pdf"):
    pickle_path = os.path.basename(pdf_path).replace(".pdf", ".pickle")
    if os.path.exists(pickle_path):
        print(f"Loading book from {pickle_path}...")
        with open(pickle_path, "rb") as f:
            book = pickle.load(f)
    else:
        print(f"Parsing book from {pdf_path}...")
        book = book.from_pdf_path(pdf_path)
        print(f"Dumping book to {pickle_path}...")
        with open(pickle_path, "wb") as f:
            pickle.dump(book, f)

    # Flatten sections
    sections = book.flatten_sections(only_leaf=True)

    # Process sections with AI
    # results = await process_sections(sections)
    # rich_view_lessons(sections, results)

    # Sort sections by importance
    start_time = datetime.now()
    sorted_sections, scores = await sort_sections(sections)
    duration = datetime.now() - start_time

    logger.info(
        "section_sorting_complete",
        duration_seconds=duration.total_seconds(),
        total_sections=len(sections),
    )

    rich_view_sorted_sections(sorted_sections, scores)
    return sorted_sections


def main(pdf_path: str = "dev/data/asyncio/asyncio.pdf"):
    asyncio.run(sort_main(pdf_path=pdf_path))


if __name__ == "__main__":
    fire.Fire(main)
