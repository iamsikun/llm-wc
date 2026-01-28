from __future__ import annotations

"""Evaluation loop and accuracy computation for benchmark datasets."""

from dataclasses import dataclass
from typing import Any, Callable, Iterable

from llm_wc.bench_utils import ProgressBar
from llm_wc.client import LLMClient
from llm_wc.client.types import ModelResponse
from llm_wc.core import BenchmarkDataset, Question


@dataclass(frozen=True)
class EvalResult:
    """Standard evaluation record returned by the evaluation layer."""

    benchmark: str  # The name of the benchmark
    question_id: int  # The question ID unique within the dataset
    original_id: str  # The question ID from the original dataset
    question: str  # The question text
    choices: dict[str, str]  # The choices for the question
    answer: str  # The correct answer to the question
    pred: str | None  # The predicted answer to the question
    prompt_type: str  # The prompt type used for the question
    model_outputs: str  # The model outputs for the question
    category: str | None  # The category of the question
    metadata: dict[str, Any]  # The metadata for the question


def filter_questions(
    bench: BenchmarkDataset,
    *,
    categories: Iterable[str] | None = None,
    limit_per_category: int | None = None,
    question_ids: Iterable[int] | None = None,
) -> list[Question]:
    """Select questions based on category, IDs, and limits.

    Args:
        bench: BenchmarkDataset to draw questions from.
        categories: Optional categories to filter by.
        limit_per_category: Optional limit per category.
        question_ids: Optional explicit list of question IDs to include.

    Returns:
        list[Question]: Filtered list of questions.
    """
    question_list: list[Question] = bench.questions

    if question_ids is not None:
        wanted = {int(qid) for qid in question_ids}
        question_list = [q for q in question_list if q.id in wanted]

    if categories:
        category_set = {str(cat) for cat in categories}

        if bench.questions_by_category is None:
            bench = bench.with_questions_by_category()

        if bench.questions_by_category:
            filtered: list[Question] = []
            for category in category_set:
                questions = bench.questions_by_category.get(category, [])
                if limit_per_category is not None:
                    questions = questions[:limit_per_category]
                filtered.extend(questions)
            question_list = filtered

    return question_list


def _extract_response_metadata(response: ModelResponse) -> dict[str, Any]:
    """Extract structured metadata from a provider response."""
    response_meta: dict[str, Any] = {}
    if response.usage is not None:
        response_meta["usage"] = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }
    if response.reasoning is not None:
        response_meta["reasoning"] = response.reasoning
    if response.finish_reason is not None:
        response_meta["finish_reason"] = response.finish_reason
    if response.model is not None:
        response_meta["response_model"] = response.model
    return response_meta


def evaluate_model_on_benchmark(
    bench: BenchmarkDataset,
    *,
    llm_client: LLMClient,
    prompt_type: str,
    prompt_builder: Callable[[BenchmarkDataset, Question, str], str],
    answer_parser: Callable[
        [str, BenchmarkDataset, Question],
        str | tuple[str | None, dict[str, Any]] | None,
    ],
    request_params: dict[str, object] | None = None,
    categories: Iterable[str] | None = None,
    limit_per_category: int | None = None,
    question_ids: Iterable[int] | None = None,
    show_progress: bool = False,
) -> list[EvalResult]:
    """Evaluate a dataset using the provided prompt builder and parser.

    Args:
        bench: BenchmarkDataset to evaluate.
        llm_client: LLMClient to reuse across calls.
        prompt_type: Prompting strategy key.
        prompt_builder: Callable that builds prompts for each question.
        answer_parser: Callable that parses model output into a prediction.
        request_params: Optional per-request parameter overrides.
        categories: Optional categories to filter by.
        limit_per_category: Optional limit per category.
        question_ids: Optional question IDs to include.
        show_progress: Whether to show a progress bar.

    Returns:
        list[EvaluationResult]: Structured evaluation results.
    """
    questions = filter_questions(
        bench,
        categories=categories,
        limit_per_category=limit_per_category,
        question_ids=question_ids,
    )

    progress = ProgressBar(len(questions), label=bench.name) if show_progress else None

    results: list[EvalResult] = []

    for question in questions:
        prompt = prompt_builder(bench, question, prompt_type)

        message = [{"role": "user", "content": prompt}]
        response = llm_client.chat(message, **(request_params or {}))

        response_text: str = response.content.replace("**", "")
        parsed = answer_parser(response_text, bench, question)

        pred: str | None
        extra_meta: dict[str, Any] = {}
        if isinstance(parsed, tuple) and len(parsed) == 2:
            pred, extra_meta = parsed
        else:
            pred = parsed  # type: ignore[assignment]

        response_meta = _extract_response_metadata(response)

        results.append(
            EvalResult(
                benchmark=bench.name,
                question_id=question.id,
                original_id=question.original_id,
                question=question.question,
                choices=question.choices,
                answer=question.answer,
                pred=pred,
                prompt_type=prompt_type,
                model_outputs=response_text,
                category=question.category,
                metadata={
                    **dict(question.metadata),
                    **extra_meta,
                    **response_meta,
                },
            )
        )

        if progress is not None:
            progress.update(1)

    if progress is not None:
        progress.close()

    return results


def compute_accuracy(
    results: list[EvalResult],
    *,
    normalizer: Callable[[str | None], str | None] | None = None,
) -> dict[str, float | int]:
    """Compute accuracy statistics for evaluation results.

    Args:
        results: Evaluation records produced by the evaluation layer.
        normalizer: Optional normalization function applied to answers.

    Returns:
        dict: Dictionary with accuracy, correct, incorrect, and total counts.
    """
    correct = 0
    incorrect = 0
    for result in results:
        pred = result.pred
        true_ans = result.answer
        if normalizer is not None:
            pred = normalizer(pred)
            true_ans = normalizer(true_ans)

        if pred is not None and true_ans is not None and pred == true_ans:
            correct += 1
        else:
            incorrect += 1

    total = correct + incorrect
    accuracy = (correct / total) if total else 0.0
    return {
        "accuracy": accuracy,
        "correct": correct,
        "incorrect": incorrect,
        "total": total,
    }
