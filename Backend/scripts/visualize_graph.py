from pathlib import Path

from ai.graph.visualize import save_graph_mermaid


OUTPUT_PATH = (
    Path(__file__).resolve().parents[1]
    / "docs"
    / "resolvex_graph.mmd"
)


def main() -> None:
    output = save_graph_mermaid(
        OUTPUT_PATH
    )

    print(
        f"ResolveX graph saved to: {output}"
    )


if __name__ == "__main__":
    main()