import json
from pathlib import Path


RESULTS_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "evaluation"
    / "generation_results_with_metrics.json"
)


with open(
    RESULTS_PATH,
    "r",
    encoding="utf-8",
) as file:
    results = json.load(file)


for result in results[:2]:

    print("=" * 100)
    print("CASE:", result["case_id"])
    print("=" * 100)

    print("\nTICKET:")
    print(result["ticket"])

    print("\nGENERATED ANSWER:")
    print(result["generated_answer"])

    print("\nEXPECTED ANSWER:")
    print(result["expected_answer"])

    print("\nREQUIRED EVIDENCE:")
    print(result["required_evidence"])

    print("\nRETRIEVED DOCUMENTS:")

    for index, document in enumerate(
        result.get("retrieved_documents", []),
        start=1,
    ):
        print(f"\n--- Document {index} ---")
        print("ID:", document.get("document_id"))
        print("Title:", repr(document.get("title")))
        print("Category:", repr(document.get("category")))
        print("Retriever:", repr(document.get("retriever")))
        print("Score:", document.get("score"))

        content = document.get(
            "content",
            "",
        )

        print(
            "Content length:",
            len(content),
        )

        print(
            "Content preview:",
            repr(content[:300]),
        )

    print("\nCURRENT METRICS:")
    print(result.get("metrics"))