"""
ocr_processor.py — OCR text extraction from images using pytesseract.
Placeholder: install Tesseract OCR engine and configure the path for production use.
"""

from app.core.logger import logger

try:
    from PIL import Image
    import pytesseract

    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    logger.warning("pytesseract / Pillow not installed – OCR will return empty strings")


def extract_text_from_image(image_path: str) -> str:
    """
    Extract text from an image file using OCR.

    Args:
        image_path: Absolute path to the image file (PNG, JPG, etc.)

    Returns:
        Extracted text string, or empty string on failure.
    """
    if not OCR_AVAILABLE:
        logger.warning(f"OCR unavailable, skipping {image_path}")
        return ""

    try:
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image, lang="eng")
        logger.debug(f"OCR extracted {len(text)} chars from {image_path}")
        return text.strip()
    except Exception as exc:
        logger.error(f"OCR failed for {image_path}: {exc}")
        return ""
