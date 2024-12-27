import fitz
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
