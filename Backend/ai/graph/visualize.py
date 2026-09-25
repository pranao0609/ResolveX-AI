from __future__ import annotations

from pathlib import Path

from ai.graph.graph import resolvex_graph


def get_graph_mermaid() -> str:
    """
    Return the ResolveX LangGraph topology as Mermaid text.
    """

    return resolvex_graph.get_graph().draw_mermaid()


def save_graph_mermaid(
    output_path: str | Path,
) -> Path:
    """
    Save the ResolveX graph topology as a Mermaid file.
    """

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    mermaid = get_graph_mermaid()

    output_path.write_text(
        mermaid,
        encoding="utf-8",
    )

    return output_path