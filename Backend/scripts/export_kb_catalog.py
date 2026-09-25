"""
export_kb_catalog.py — Export the current Knowledge Base
catalog for retrieval evaluation.

Exports only metadata needed to construct an evaluation benchmark.
No model-generated relevance labels are created here.
"""

import json
import os
import sys
from pathlib import Path

# Ensure Backend directory is on Python path
BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))

from app.core.logger import logger
from app.database import SessionLocal, init_db
from app.models.kb_model import KnowledgeBaseEntry


OUTPUT_PATH = (
    BACKEND_ROOT
    / "data"
    / "evaluation"
    / "kb_catalog.json"
)


def export_kb_catalog() -> None:
    """Export Knowledge Base metadata to JSON."""

    init_db()

    db = SessionLocal()

    try:
        entries = (
            db.query(KnowledgeBaseEntry)
            .order_by(KnowledgeBaseEntry.id)
            .all()
        )

        catalog = []

        for entry in entries:
            catalog.append(
                {
                    "document_id": f"kb:{entry.id}",
                    "title": entry.title,
                    "category": entry.category,
                    "source": entry.source,
                    "faiss_index_id": entry.faiss_index_id,
                }
            )

        OUTPUT_PATH.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with open(
            OUTPUT_PATH,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                catalog,
                file,
                ensure_ascii=False,
                indent=2,
            )

        logger.info(
            "KB catalog exported successfully: %d entries",
            len(catalog),
        )

        logger.info(
            "Catalog path: %s",
            OUTPUT_PATH,
        )

    finally:
        db.close()


if __name__ == "__main__":
    export_kb_catalog()