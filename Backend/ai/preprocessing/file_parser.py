"""
file_parser.py — Unified file content extractor.
Routes each file type to the appropriate extraction method (OCR, PDF, plain text).
"""

import os
from typing import List

from app.core.logger import logger
from ai.preprocessing.ocr_processor import extract_text_from_image

try:
    import PyPDF2

    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    logger.warning("PyPDF2 not installed – PDF parsing will return empty strings")


SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".gif"}
SUPPORTED_TEXT_EXTENSIONS = {".txt", ".log", ".csv"}


def parse_attachments(file_paths: List[str]) -> str:
    """
    Extract and concatenate text from a list of file paths.

    Args:
        file_paths: List of absolute or relative paths to attachment files.

    Returns:
        Combined extracted text from all files.
    """
    extracted_parts: List[str] = []

    for path in file_paths:
        if not os.path.exists(path):
            logger.warning(f"Attachment not found: {path}")
            continue

        ext = os.path.splitext(path)[1].lower()
        logger.debug(f"Parsing attachment: {path} (ext={ext})")

        if ext in SUPPORTED_IMAGE_EXTENSIONS:
            text = extract_text_from_image(path)
        elif ext == ".pdf":
            text = _extract_pdf_text(path)
        elif ext in SUPPORTED_TEXT_EXTENSIONS:
            text = _read_plain_text(path)
        else:
            logger.warning(f"Unsupported file type: {ext} for {path}")
            text = ""

        if text:
            extracted_parts.append(f"[{os.path.basename(path)}]\n{text}")

    return "\n\n".join(extracted_parts)


def _extract_pdf_text(path: str) -> str:
    """Extract text from a PDF file using PyPDF2."""
    if not PDF_AVAILABLE:
        return ""
    try:
        text_parts = []
        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text_parts.append(page.extract_text() or "")
        return "\n".join(text_parts).strip()
    except Exception as exc:
        logger.error(f"PDF extraction failed for {path}: {exc}")
        return ""


def _read_plain_text(path: str) -> str:
    """Read a plain text / log file."""
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read().strip()
    except Exception as exc:
        logger.error(f"Text read failed for {path}: {exc}")
        return ""
