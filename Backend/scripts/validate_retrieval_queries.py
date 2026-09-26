"""
validate_retrieval_queries.py — Validate retrieval evaluation
queries against the current ResolveX KB catalog.
"""

import json
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]

CATALOG_PATH = BACKEND_ROOT / "data" / "evaluation" / "kb_catalog.json"

QUERIES_PATH = BACKEND_ROOT / "data" / "evaluation" / "retrieval_queries.json"


def main():
    with open(
        CATALOG_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        catalog = json.load(file)

    with open(
        QUERIES_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        queries = json.load(file)

    valid_document_ids = {entry["document_id"] for entry in catalog}

    print("=" * 70)
    print("RETRIEVAL EVALUATION DATASET VALIDATION")
    print("=" * 70)

    print(f"KB documents: {len(catalog)}")
    print(f"Evaluation queries: {len(queries)}")

    errors = []

    query_ids = set()

    for item in queries:
        query_id = item.get("query_id")
        query = item.get("query")
        relevant_ids = item.get("relevant_document_ids")

        if not query_id:
            errors.append("Missing query_id")
            continue

        if query_id in query_ids:
            errors.append(f"Duplicate query_id: {query_id}")

        query_ids.add(query_id)

        if not isinstance(query, str) or not query.strip():
            errors.append(f"{query_id}: invalid query")

        if not isinstance(relevant_ids, list):
            errors.append(f"{query_id}: relevant_document_ids must be a list")
            continue

        if not relevant_ids:
            errors.append(f"{query_id}: no relevant documents")

        for document_id in relevant_ids:
            if document_id not in valid_document_ids:
                errors.append(f"{query_id}: unknown document ID {document_id}")

    print()

    if errors:
        print(f"VALIDATION FAILED: {len(errors)} errors")

        for error in errors:
            print(f"- {error}")

        raise SystemExit(1)

    print("VALIDATION PASSED")
    print(f"Queries validated: {len(queries)}")
    print("All relevant document IDs exist in the KB catalog.")
    print("=" * 70)


if __name__ == "__main__":
    main()
