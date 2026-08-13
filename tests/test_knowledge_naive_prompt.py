from src.eval.tasks.knowledge.runner import _naive_direct_prompt_template


def test_naive_direct_prompt_prefills_complete_empty_think() -> None:
    prompt = _naive_direct_prompt_template()
    assert prompt.endswith("Assistant: <think></think>")
    assert "The answer is" not in prompt
