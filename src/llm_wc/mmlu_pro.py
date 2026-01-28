from __future__ import annotations

"""MMLU-Pro evaluation helpers."""

from typing import Any

from datasets import Dataset, load_dataset
from llm_wc.core.dataset import BenchmarkDataset, Question
from llm_wc.core.questions import DEFAULT_CHOICES
from llm_wc.core.eval import EvalResult, evaluate_model_on_benchmark
from llm_wc.client import LLMClient
from llm_wc.core.mcqa import extract_choice_answer


MMLU_PRO_DATASET = "TIGER-Lab/MMLU-Pro"
MMLU_PRO_PROMPT = (
    "The following are multiple choice questions (with answers) about {category}. "
    'Think step by step and then output the answer in the format of "The answer is (X)" at the end.'
)


def _preprocess(ds: Dataset) -> list[dict]:
    """
    Remove N/A options and return processed question dicts.

    Args:
        ds: The dataset to preprocess.

    Returns:
        A list of processed question dicts.
    """
    questions: list[dict] = []
    for question_dict in ds:
        options = [opt for opt in question_dict["options"] if opt != "N/A"]
        question_dict = dict(question_dict)
        question_dict["options"] = options
        questions.append(question_dict)
    return questions


def _build_questions(questions: list[dict[str, Any]]) -> list[Question]:
    """
    Build Question instances from the raw data.

    Args:
        questions: The raw data to build questions from.

    Returns:
        A list of Question instances.
    """
    processed_questions: list[Question] = []

    # sort questions by the original question id
    questions.sort(key=lambda x: x.get("question_id", 0))

    # build Question instances
    for idx, item in enumerate(questions):
        options: list[str] = item.get("options", [])
        choices: dict[str, str] = {
            DEFAULT_CHOICES[i]: str(option) for i, option in enumerate(options)
        }
        original_id = str(item.get("question_id", idx))
        processed_questions.append(
            Question(
                id=idx,
                original_id=original_id,
                question=str(item.get("question", "")),
                choices=choices,
                answer=str(item.get("answer", "")),
                category=item.get("category"),
                metadata={
                    "source": item.get("source"),
                },
            )
        )
    return processed_questions


def _build_cot_examples(raw: list[dict]) -> dict[str, list[dict]]:
    """Group COT examples by category for prompt assembly."""
    cot_by_category: dict[str, list[dict]] = {}
    for item in raw:
        category = item.get("category")
        if category is None:
            continue
        cot_by_category.setdefault(category, []).append(item)
    return cot_by_category


def load_mmlu_pro_benchmark() -> BenchmarkDataset:
    """
    Load the MMLU-Pro dataset and return a BenchmarkDataset instance.

    Returns:
        A BenchmarkDataset instance.
    """
    # load datasets from HuggingFace
    dataset: Dataset = load_dataset(MMLU_PRO_DATASET)

    # preprocess the datasets
    test_questions: list[dict[str, Any]] = _preprocess(dataset["test"])
    val_questions: list[dict[str, Any]] = _preprocess(dataset["validation"])

    # build questions
    questions: list[Question] = _build_questions(test_questions)

    # build cot examples
    cot_examples: dict[str, list[dict]] = _build_cot_examples(val_questions)

    # build benchmark dataset
    bench_ds: BenchmarkDataset = BenchmarkDataset(
        name="mmlu_pro",
        questions=questions,
        prompts={
            "default": MMLU_PRO_PROMPT,
            "cot": MMLU_PRO_PROMPT,
        },
        metadata={"cot_examples_by_category": cot_examples},
        categories=list(cot_examples.keys()),
    )

    return bench_ds


def _choices_to_list(choices: dict[str, str]) -> list[str]:
    """Convert A-J choice mapping into an ordered list."""
    ordered = []
    for letter in DEFAULT_CHOICES[:10]:
        if letter in choices:
            ordered.append(choices[letter])
    return ordered


def format_example(question: str, options: list[str], cot_content: str = "") -> str:
    """Format an example for the MMLU-Pro dataset."""
    if cot_content == "":
        cot_content = "Let's think step by step."
    if cot_content.startswith("A: "):
        cot_content = cot_content[3:]

    example = f"Question: {question}\nOptions: "
    choice_map = DEFAULT_CHOICES[:10]
    for i, opt in enumerate(options):
        example += f"{choice_map[i]}. {opt}\n"

    if cot_content == "":
        example += "Answer: "
    else:
        example += "Answer: " + cot_content + "\n\n"
    return example


def extract_answer(text: str) -> str | None:
    """Extract the answer from the text using shared MCQ parsing utilities."""
    result = extract_choice_answer(text, choices=DEFAULT_CHOICES[:10])
    return result.choice


def prompt_mmlu_pro(
    bench: BenchmarkDataset, question: Question, prompt_type: str
) -> str:
    """Build the MMLU-Pro prompt with COT examples and target question."""
    category = question.category or "unknown"
    instruction = bench.prompts.get(prompt_type, MMLU_PRO_PROMPT).format(
        category=category
    )
    prompt = instruction + "\n\n"

    cot_examples = bench.metadata.get("cot_examples_by_category", {}).get(category, [])
    for cot_example in cot_examples:
        prompt += format_example(
            cot_example["question"],
            cot_example["options"],
            cot_example.get("cot_content", ""),
        )

    input_text = format_example(question.question, _choices_to_list(question.choices))
    return prompt + input_text


def evaluate_model_on_mmlu_pro(
    *,
    llm_client: LLMClient,
    subjects: list[str] | None = None,
    limit_per_subject: int | None = None,
    prompt_type: str = "cot",
    request_params: dict[str, object] | None = None,
    show_progress: bool = False,
) -> list[EvalResult]:
    """
    Evaluate the model on the MMLU-Pro dataset and return evaluation results.

    Args:
        llm_client: LLMClient to reuse across calls.
        subjects: Optional subjects to filter by.
        limit_per_subject: Optional limit per subject.
        prompt_type: Prompt type to use.
        request_params: Optional per-request parameter overrides.
        show_progress: Whether to show a progress bar.

    Returns:
        list[EvalResult]: Structured evaluation results.
    """
    # load benchmark
    bench = load_mmlu_pro_benchmark()

    # build prompt builder
    def prompt_builder(
        bench: BenchmarkDataset, question: Question, prompt_key: str
    ) -> str:
        return prompt_mmlu_pro(bench, question, prompt_key)

    def answer_parser(
        text: str, bench: BenchmarkDataset, question: Question
    ) -> str | None:
        return extract_answer(text)

    # evaluate benchmark
    return evaluate_model_on_benchmark(
        bench,
        llm_client=llm_client,
        prompt_type=prompt_type,
        prompt_builder=prompt_builder,
        answer_parser=answer_parser,
        request_params=request_params,
        categories=subjects,
        limit_per_category=limit_per_subject,
        show_progress=show_progress,
    )
