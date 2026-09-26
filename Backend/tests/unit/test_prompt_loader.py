from ai.llm.prompt_loader import load_prompt, render_prompt


def test_load_resolution_prompt_v1():
    prompt = load_prompt("resolution", 1)

    assert prompt["prompt_id"] == "resolution_prompt_v1"
    assert prompt["version"] == 1
    assert prompt["model"]
    assert prompt["temperature"] == 0.3
    assert prompt["created_at"]
    assert prompt["system_prompt"]
    assert prompt["user_prompt"]


def test_load_resolution_prompt_v2():
    prompt = load_prompt("resolution", 2)

    assert prompt["prompt_id"] == "resolution_prompt_v2"
    assert prompt["version"] == 2


def test_load_resolution_prompt_v3():
    prompt = load_prompt("resolution", 3)

    assert prompt["prompt_id"] == "resolution_prompt_v3"
    assert prompt["version"] == 3


def test_render_prompt():
    prompt = load_prompt("resolution", 1)

    system_prompt, user_prompt = render_prompt(
        prompt,
        ticket_text="Email login is failing.",
        context="User Cannot Access Email",
    )

    assert system_prompt
    assert "Email login is failing." in user_prompt
    assert "User Cannot Access Email" in user_prompt
