import json
from pathlib import Path


PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "evaluation"
    / "retrieval_results.json"
)


with open(
    PATH,
    "r",
    encoding="utf-8",
) as file:
    data = json.load(file)


print("=" * 100)
print("RETRIEVAL RESULTS STRUCTURE")
print("=" * 100)

print("\nTop-level type:")
print(type(data).__name__)

if isinstance(data, dict):

    print("\nTop-level keys:")
    for key in data.keys():
        print(f"  {key}")

    print("\nTop-level values:")
    for key, value in data.items():

        if isinstance(value, list):
            print(
                f"  {key}: list "
                f"(length={len(value)})"
            )

        elif isinstance(value, dict):
            print(
                f"  {key}: dict "
                f"(keys={list(value.keys())[:10]})"
            )

        else:
            print(
                f"  {key}: "
                f"{type(value).__name__} = {value}"
            )

elif isinstance(data, list):

    print(
        f"\nList length: {len(data)}"
    )

    if data:
        print("\nFirst item:")
        print(
            json.dumps(
                data[0],
                indent=2,
            )[:5000]
        )