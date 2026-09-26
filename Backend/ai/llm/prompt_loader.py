from pathlib import Path

import yaml

PROMPTS_DIR = Path(__file__).resolve().parents[2] / "prompts"


def load_prompt(prompt_type: str, version: int) -> dict:
    """
    Load a versioned prompt definition.

    Example:
        load_prompt("resolution", 1)
    """

    prompt_path = PROMPTS_DIR / prompt_type / f"{prompt_type}_prompt_v{version}.yaml"

    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt not found: {prompt_path}")

    with prompt_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        prompt = yaml.safe_load(file)

    required_fields = {
        "prompt_id",
        "version",
        "model",
        "temperature",
        "created_at",
        "system_prompt",
        "user_prompt",
    }

    missing = required_fields - set(prompt.keys())

    if missing:
        raise ValueError(
            f"Prompt {prompt_path} is missing fields: " f"{sorted(missing)}"
        )

    return prompt


def render_prompt(
    prompt: dict,
    ticket_text: str,
    context: str,
) -> tuple[str, str]:
    """
    Render a versioned prompt using runtime ticket/context values.

    Uses explicit placeholder replacement instead of str.format()
    so JSON braces inside prompt templates remain untouched.
    """

    system_prompt = prompt["system_prompt"]

    user_prompt = prompt["user_prompt"]

    user_prompt = user_prompt.replace(
        "{ticket_text}",
        ticket_text,
    )

    user_prompt = user_prompt.replace(
        "{context}",
        context,
    )

    return system_prompt, user_prompt
