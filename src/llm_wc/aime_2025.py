from __future__ import annotations

"""AIME 2025 evaluation helpers."""

from typing import Iterable

from llm_wc.core import BenchmarkDataset, Question
from llm_wc.core import EvalResult, evaluate_model_on_benchmark
from llm_wc.matharena import (
    load_matharena_competition_problems,
    ParsedAnswer,
    extract_answer,
)
from llm_wc.client import LLMClient

# Ollama OpenAI-compatible endpoint
OLLAMA_URL = "http://localhost:11434/v1"

# Must match an Ollama model name from `ollama list`
MODEL_NAME = "glm-4.7-flash:latest"

# Dataset paths and prompts
AIME_2025_DATASET = "MathArena/aime_2025"
AIME_2025_PROMPT = (
    "Please reason step by step, and put your final answer within \\boxed{}.\n"
    "The answer is an integer between 0 and 999 inclusive."
)


def load_aime_2025_benchmark(
    *, dataset_path: str = AIME_2025_DATASET
) -> BenchmarkDataset:
    """Load AIME 2025 into the shared Dataset structure."""
    problems = load_matharena_competition_problems(
        dataset_path, problem_ids=None, final_answer=True
    )

    questions: list[Question] = []
    for idx, problem in enumerate(problems):
        original_id = str(problem.get("problem_idx", idx))
        problem_type = problem.get("problem_type")
        questions.append(
            Question(
                id=idx,
                original_id=original_id,
                question=str(problem.get("problem", "")),
                choices={},
                answer=str(problem.get("answer", "")),
                category=problem_type if isinstance(problem_type, str) else None,
                metadata={
                    "image": problem.get("image"),
                    "problem_type": problem_type,
                    "source": problem.get("source"),
                },
            )
        )

    return BenchmarkDataset(
        name="aime_2025",
        questions=questions,
        prompts={"default": AIME_2025_PROMPT, "zero_shot": AIME_2025_PROMPT},
        metadata={"dataset_path": dataset_path},
    )


def normalize_aime_answer(value: str | int | None) -> int | None:
    """Normalize an AIME answer into an integer when possible."""
    if value is None:
        return None
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if text.isdigit():
        return int(text)
    return None


def prompt_aime_2025(
    dataset: BenchmarkDataset, question: Question, prompt_type: str
) -> str:
    """Construct the prompt sent to the model for an AIME problem."""
    instruction = dataset.prompts.get(prompt_type, AIME_2025_PROMPT)
    problem_text = question.question
    if not problem_text:
        return instruction + "\n\n[Problem statement missing from dataset.]"
    return instruction + "\n\n" + problem_text


def _parse_prediction(text: str, *, strict_parsing: bool) -> ParsedAnswer:
    """Parse the model output into an AIME-compatible answer."""
    return extract_answer(text, strict_parsing=strict_parsing)


def evaluate_model_on_aime_2025(
    *,
    llm_client: LLMClient,
    problem_ids: Iterable[int] | None = None,
    limit: int | None = None,
    dataset_path: str = AIME_2025_DATASET,
    instruction: str = AIME_2025_PROMPT,
    strict_parsing: bool = False,
    request_params: dict[str, object] | None = None,
    show_progress: bool = False,
) -> list[EvalResult]:
    """Evaluate a model on the AIME 2025 benchmark and return evaluation results."""
    dataset = load_aime_2025_benchmark(dataset_path=dataset_path)
    dataset = BenchmarkDataset(
        name=dataset.name,
        questions=dataset.questions,
        prompts={"default": instruction},
        description=dataset.description,
        metadata=dataset.metadata,
        categories=dataset.categories,
        questions_by_category=dataset.questions_by_category,
    )
    if limit is not None:
        dataset = BenchmarkDataset(
            name=dataset.name,
            questions=dataset.questions[:limit],
            prompts=dataset.prompts,
            description=dataset.description,
            metadata=dataset.metadata,
            categories=dataset.categories,
            questions_by_category=dataset.questions_by_category,
        )

    def prompt_builder(
        ds: BenchmarkDataset, question: Question, prompt_key: str
    ) -> str:
        return prompt_aime_2025(ds, question, prompt_key)

    def answer_parser(
        text: str, ds: BenchmarkDataset, question: Question
    ) -> tuple[str | None, dict[str, str]]:
        parsed = _parse_prediction(text, strict_parsing=strict_parsing)
        return parsed.answer, {"warning": parsed.warning.name}

    return evaluate_model_on_benchmark(
        dataset,
        llm_client=llm_client,
        prompt_type="default",
        prompt_builder=prompt_builder,
        answer_parser=answer_parser,
        request_params=request_params,
        question_ids=problem_ids,
        show_progress=show_progress,
    )
