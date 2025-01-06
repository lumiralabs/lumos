import structlog
import httpx
import json
import time
import os
from ..utils import file_cache_middleware

logger = structlog.get_logger()


def get_section_text_map(md_file: str) -> dict[str, str]:
    current_title = ""
    current_text = []
    section2text = {}
    with open(md_file, "r") as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        line = line.strip()
        if line.startswith("#"):
            # If we were tracking a previous section, save it
            if current_title:
                section2text[current_title] = "\n".join(current_text).strip()
                current_text = []

            current_title = line.lstrip("#").strip()
        else:
            current_text.append(line)

    # Handle last section
    if current_title:
        section2text[current_title] = "\n".join(current_text).strip()

    return section2text


@file_cache_middleware
def mathpix_pdf_to_markdown(pdf_path: str) -> str:
    """
    Convert a PDF file to markdown using Mathpix API.
    Results are automatically cached based on the file content.
    """
    options = {
        "conversion_formats": {"md": True},
        "math_inline_delimiters": ["$", "$"],
        "rm_spaces": True,
    }

    if not (APP_ID := os.getenv("MATHPIX_APP_ID")):
        raise ValueError("MATHPIX_APP_ID is not set")

    if not (APP_KEY := os.getenv("MATHPIX_APP_KEY")):
        raise ValueError("MATHPIX_APP_KEY is not set")

    # Submit PDF for conversion
    logger.info("[mathpix] submitting_pdf_for_conversion", pdf_path=pdf_path)
    r = httpx.post(
        "https://api.mathpix.com/v3/pdf",
        headers={"app_id": APP_ID, "app_key": APP_KEY},
        data={"options_json": json.dumps(options)},
        files={"file": open(pdf_path, "rb")},
    )

    pdf_id = r.json()["pdf_id"]
    logger.info("[mathpix] pdf_submitted", pdf_id=pdf_id)

    # Poll until conversion is complete
    while True:
        response = httpx.get(
            f"https://api.mathpix.com/v3/pdf/{pdf_id}",
            headers={"app_id": APP_ID, "app_key": APP_KEY},
        )
        status = response.json()

        if status["status"] == "completed":
            break
        elif status["status"] != "completed":
            logger.info("[mathpix] conversion_status", status=status["status"])

        time.sleep(2)

    # Get markdown output
    logger.info("[mathpix] downloading_markdown")
    response = httpx.get(
        f"https://api.mathpix.com/v3/pdf/{pdf_id}.md",
        headers={"app_id": APP_ID, "app_key": APP_KEY},
    )

    return response.text


def convert_pdf_to_markdown(pdf_path: str) -> str:
    """
    Convert a PDF file to markdown using Mathpix API.

    Args:
        pdf_path: Path to the PDF file
        app_id: Mathpix app ID
        app_key: Mathpix app key

    Returns:
        Path to the generated markdown file
    """

    markdown_text = mathpix_pdf_to_markdown(pdf_path)

    output_path = f"{pdf_path}_mathpix.md"
    with open(output_path, "w") as f:
        f.write(markdown_text)

    logger.info("markdown_saved", output_path=output_path)
    return output_path
