import inspect

from evaluation.generation.evaluator import GenerationEvaluator


def test_evaluate_case_accepts_prompt_version():
    signature = inspect.signature(GenerationEvaluator.evaluate_case)

    assert "prompt_version" in signature.parameters
    assert signature.parameters["prompt_version"].default == 1


def test_evaluate_all_accepts_prompt_version():
    signature = inspect.signature(GenerationEvaluator.evaluate_all)

    assert "prompt_version" in signature.parameters
    assert signature.parameters["prompt_version"].default == 1


def test_save_results_requires_output_path():
    signature = inspect.signature(GenerationEvaluator.save_results)

    assert "output_path" in signature.parameters
    assert signature.parameters["output_path"].default is inspect.Parameter.empty
