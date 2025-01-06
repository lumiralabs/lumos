from pdf2image import convert_from_path
import fitz
import io
from .models import PDFMetadata


def extract_pdf_metadata(pdf_path: str) -> PDFMetadata:
    """Extract metadata from a PDF file."""
    with fitz.open(pdf_path) as doc:
        raw_title = doc.metadata.get("title", "")
        raw_author = doc.metadata.get("author", "")
        raw_subject = doc.metadata.get("subject")
        raw_keywords = doc.metadata.get("keywords", "")
        keywords = raw_keywords.split(",") if raw_keywords else None

        return PDFMetadata(
            title=raw_title,
            author=raw_author,
            subject=raw_subject,
            keywords=keywords,
            path=pdf_path,
            toc=doc.get_toc(),
        )


def extract_pdf_pages_as_images(pdf_path: str, page_numbers: list[int]) -> list[bytes]:
    """
    Extracts the given 1-based page numbers from a PDF as JPG images.

    :param pdf_path: The path to the PDF file.
    :param page_numbers: 1-based page numbers to extract.
    :return: A list of bytes (each representing a single page image in JPG format).
    """
    images_as_bytes = []
    for page_number in page_numbers:
        # pdf2image expects 1-based pages, so we can use page_number directly
        pages = convert_from_path(
            pdf_path,
            dpi=100,
            first_page=page_number,
            last_page=page_number,
        )
        if pages:
            # Typically one image per page.
            image = pages[0]
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG")
            buffer.seek(0)
            images_as_bytes.append(buffer.read())
    return images_as_bytes
