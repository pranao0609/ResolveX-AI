"""
export_kb_evaluation_dataset.py

Export the current ResolveX knowledge base into a JSON dataset
for RAG generation evaluation.

The exported dataset contains the actual KB content used by
the RAG pipeline.
"""

import json
import sys
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(BACKEND_ROOT))


from app.database import SessionLocal
from app.models.kb_model import KnowledgeBaseEntry


OUTPUT_PATH = (
    BACKEND_ROOT
    / "data"
    / "evaluation"
    / "kb_evaluation_dataset.json"
)


def main():
    """Export knowledge-base entries for evaluation."""

    db = SessionLocal()

    try:
        entries = (
            db.query(KnowledgeBaseEntry)
            .order_by(KnowledgeBaseEntry.id.asc())
            .all()
        )

        dataset = []

        for entry in entries:
            dataset.append(
                {
                    "document_id": f"kb:{entry.id}",
                    "title": entry.title,
                    "category": entry.category,
                    "source": entry.source,
                    "content": entry.content,
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
                dataset,
                file,
                indent=2,
                ensure_ascii=False,
            )

        print("=" * 80)
        print("RESOLVEX KB EVALUATION DATASET EXPORT")
        print("=" * 80)
        print(f"Knowledge-base entries: {len(dataset)}")
        print(f"Output: {OUTPUT_PATH}")
        print("=" * 80)

    finally:
        db.close()


if __name__ == "__main__":
    main()