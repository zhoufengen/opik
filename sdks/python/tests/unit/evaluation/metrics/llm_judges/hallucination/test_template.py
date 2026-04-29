from opik.evaluation.metrics.llm_judges.hallucination.template import (
    FewShotExampleHallucination,
    generate_query,
)


_EXAMPLE: FewShotExampleHallucination = {
    "title": "ex1",
    "input": "What is the capital of France?",
    "context": ["France is a country in Europe."],
    "output": "Paris is the capital of France.",
    "score": 0.0,
    "reason": "factual",
}


def test_few_shot_with_context_renders_full_example():
    rendered = generate_query(
        input="q",
        output="a",
        context=["c"],
        few_shot_examples=[_EXAMPLE],
    )

    assert "<example>" in rendered
    assert "Input: What is the capital of France?" in rendered
    assert "Context: ['France is a country in Europe.']" in rendered
    assert "Output: Paris is the capital of France." in rendered
    assert '"score": "0.0"' in rendered
    assert '"reason": "factual"' in rendered
    assert "</example>" in rendered


def test_few_shot_without_context_renders_full_example():
    rendered = generate_query(
        input="q",
        output="a",
        context=None,
        few_shot_examples=[_EXAMPLE],
    )

    assert "<example>" in rendered
    assert "Input: What is the capital of France?" in rendered
    assert "Output: Paris is the capital of France." in rendered
    assert '"score": "0.0"' in rendered
    assert '"reason": "factual"' in rendered
    assert "</example>" in rendered


def test_multiple_few_shot_examples_separated():
    second: FewShotExampleHallucination = {
        "title": "ex2",
        "input": "Who wrote Hamlet?",
        "context": ["Shakespeare authored many plays."],
        "output": "Shakespeare wrote Hamlet.",
        "score": 0.0,
        "reason": "correct",
    }

    rendered = generate_query(
        input="q",
        output="a",
        context=["c"],
        few_shot_examples=[_EXAMPLE, second],
    )

    assert rendered.count("<example>") == 2
    assert rendered.count("</example>") == 2
    assert "Who wrote Hamlet?" in rendered
    assert "Shakespeare wrote Hamlet." in rendered
