"""
file_manager.py — High-level file management abstraction.
Delegates actual storage to local_storage or cloud adapters.
"""

import os
from fastapi import UploadFile
from app.core.logger import logger
from app.config import settings
from storage.local_storage import LocalStorage


class FileManager:
    """
    Facade over storage backends (local disk, S3, etc.).
    Currently backed by LocalStorage; swap out for cloud storage in production.
    """

    def __init__(self):
        self.backend = LocalStorage(base_dir=settings.UPLOAD_DIR)

    async def save(self, file: UploadFile) -> str:
        """
        Save an uploaded file and return its stored path.

        Args:
            file: FastAPI UploadFile object

        Returns:
            Absolute or relative path to the saved file.
        """
        logger.info(f"Saving file: {file.filename}")
        content = await file.read()
        path = self.backend.write(filename=file.filename, content=content)
        logger.debug(f"File saved to: {path}")
        return path

    def delete(self, path: str) -> None:
        """Remove a file from storage."""
        self.backend.remove(path)

    def exists(self, path: str) -> bool:
        return self.backend.exists(path)


# Global instance for __init__.py import (used throughout app)
file_manager = FileManager()
