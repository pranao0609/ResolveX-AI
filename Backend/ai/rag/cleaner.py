"""
cleaner.py — Knowledge-base document cleaning for RAG ingestion.

This cleaner is intentionally separate from ticket preprocessing.

RAG documents preserve:
- paragraph boundaries
- technical capitalization
- commands
- URLs
- file paths
- error messages
- meaningful punctuation
"""

import re

from ai.rag.models import CanonicalDocument


class DocumentCleaner:
    """Clean and normalize knowledge-base documents for RAG."""

    @staticmethod
    def clean(document: CanonicalDocument) -> CanonicalDocument:
        """
        Return a cleaned copy of a canonical document.

        The original document is not mutated.
        """

        title = DocumentCleaner._clean_title(document.title)
        content = DocumentCleaner._clean_content(document.content)

        return document.model_copy(
            update={
                "title": title,
                "content": content,
            }
        )

    @staticmethod
    def _clean_title(title: str) -> str:
        """Normalize document title while preserving capitalization."""

        if not title:
            return ""

        # Remove HTML tags.
        title = re.sub(r"<[^>]+>", " ", title)

        # Collapse whitespace in titles.
        title = re.sub(r"[ \t\r\n]+", " ", title)

        return title.strip()

    @staticmethod
    def _clean_content(content: str) -> str:
        """
        Normalize document content while preserving paragraph structure.
        """

        if not content:
            return ""

        # Remove HTML tags without lowercasing technical content.
        content = re.sub(r"<[^>]+>", " ", content)

        # Normalize Windows/macOS line endings.
        content = content.replace("\r\n", "\n").replace("\r", "\n")

        # Normalize tabs to spaces.
        content = content.replace("\t", " ")

        # Remove trailing spaces from each line.
        lines = [line.rstrip() for line in content.split("\n")]

        # Collapse runs of spaces inside individual lines.
        lines = [re.sub(r" {2,}", " ", line) for line in lines]

        # Remove leading/trailing whitespace from individual lines.
        lines = [line.strip() for line in lines]

        # Preserve paragraph boundaries, but avoid excessive blank lines.
        cleaned_lines = []

        previous_blank = False

        for line in lines:
            is_blank = not line

            if is_blank:
                if not previous_blank:
                    cleaned_lines.append("")
                previous_blank = True
            else:
                cleaned_lines.append(line)
                previous_blank = False

        return "\n".join(cleaned_lines).strip()