import inspect

from ai.pipeline.ticket_pipeline import run_pipeline


def test_run_pipeline_accepts_prompt_version():
    signature = inspect.signature(run_pipeline)

    assert "prompt_version" in signature.parameters
    assert signature.parameters["prompt_version"].default == 1
