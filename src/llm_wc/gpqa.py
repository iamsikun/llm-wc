from __future__ import annotations

"""GPQA benchmark loading and evaluation helpers."""

import csv
import json
import random
from pathlib import Path
from typing import Iterable

from llm_wc.client import LLMClient
from llm_wc.core import BenchmarkDataset, Question
from llm_wc.core.eval import EvalResult, evaluate_model_on_benchmark
from llm_wc.core.mcqa import ChoiceAnswer, extract_choice_answer
from llm_wc.hf_utils import is_gated_dataset_error, load_hf_dataset_rows

GPQA_DATASET = "idavidrein/gpqa"
GPQA_DEFAULT_SPLIT = "train"
GPQA_DEFAULT_PROMPT_TYPE = "zero_shot"
GPQA_DEFAULT_COT_EXAMPLES = "resources/gpqa/chain_of_thought_examples.json"

GPQA_CHOICES = ["A", "B", "C", "D"]

GPQA_PROMPTS = {
    "zero_shot": 'Format your response as follows: "The correct answer is (insert answer here)"',
    "chain_of_thought": (
        "Answer the final question yourself, giving your reasoning beforehand."
    ),
    "5_shot": (
        "Answer the final question yourself, following the format of the previous questions exactly."
    ),
}


def _normalize_row_keys(row: dict) -> dict:
    """Normalize differing GPQA CSV column headers into a common schema."""
    lowered = {str(key).strip().lower(): value for key, value in row.items()}

    def first(keys: Iterable[str]) -> str | None:
        for key in keys:
            if key in row:
                return row[key]
            if key.lower() in lowered:
                return lowered[key.lower()]
        return None

    return {
        "question": first(["Question", "question"]),
        "correct": first(["Correct Answer", "correct_answer", "correct"]),
        "incorrect_1": first(
            ["Incorrect Answer 1", "incorrect_answer_1", "incorrect_1"]
        ),
        "incorrect_2": first(
            ["Incorrect Answer 2", "incorrect_answer_2", "incorrect_2"]
        ),
        "incorrect_3": first(
            ["Incorrect Answer 3", "incorrect_answer_3", "incorrect_3"]
        ),
        "question_id": first(["question_id", "id", "QuestionID"]),
    }


def _rows_from_csv(path: Path) -> list[dict]:
    """Read GPQA rows from a local CSV file."""
    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def _gated_dataset_help(
    *, dataset_path: str, subset: str | None, split: str, hf_token_env: str | None
) -> str:
    env_hint = hf_token_env or "HF_TOKEN"
    subset_hint = f", subset={subset!r}" if subset else ""
    return (
        "GPQA dataset access failed because the Hugging Face dataset is gated.\n\n"
        "Fix options:\n"
        "1) Authenticate to Hugging Face and accept access for the dataset.\n"
        f"   - Set an access token in an env var (recommended): export {env_hint}=<your_hf_token>\n"
        "   - Or run: huggingface-cli login\n\n"
        "2) Use a local CSV instead of the Hub:\n"
        "   - Download the GPQA CSVs locally and set benchmark.dataset_path to either:\n"
        "     - a directory containing gpqa_main.csv / gpqa_diamond.csv, or\n"
        "     - a direct path to the CSV file.\n\n"
        f"Details: dataset_path={dataset_path!r}{subset_hint}, split={split!r}"
    )


def _rows_from_dataset(
    dataset_path: str,
    subset: str | None,
    split: str,
    *,
    hf_token: str | None = None,
    hf_token_env: str | None = None,
) -> list[dict]:
    """Load GPQA rows from Hugging Face with gated-dataset guidance."""
    try:
        return load_hf_dataset_rows(
            dataset_path,
            subset=subset,
            split=split,
            hf_token=hf_token,
            hf_token_env=hf_token_env,
        )
    except Exception as exc:
        if is_gated_dataset_error(exc):
            raise RuntimeError(
                _gated_dataset_help(
                    dataset_path=dataset_path,
                    subset=subset,
                    split=split,
                    hf_token_env=hf_token_env,
                )
            ) from exc
        raise


def load_gpqa_benchmark(
    *,
    dataset_path: str = GPQA_DATASET,
    subset: str | None = None,
    split: str = GPQA_DEFAULT_SPLIT,
    seed: int = 0,
    max_examples: int | None = None,
    hf_token: str | None = None,
    hf_token_env: str | None = None,
) -> BenchmarkDataset:
    """Load GPQA into the shared Dataset structure."""
    path = Path(dataset_path)
    if path.exists() and path.is_dir():
        if subset:
            local_subset = subset
            if local_subset.startswith("gpqa_"):
                local_subset = local_subset[len("gpqa_") :]
            csv_path = path / f"gpqa_{local_subset}.csv"
        else:
            csv_path = path / "gpqa_main.csv"
        rows = _rows_from_csv(csv_path)
    elif path.exists() and path.is_file():
        rows = _rows_from_csv(path)
    else:
        rows = _rows_from_dataset(
            dataset_path,
            subset,
            split,
            hf_token=hf_token,
            hf_token_env=hf_token_env,
        )

    rng = random.Random(seed)
    questions: list[Question] = []
    for idx, row in enumerate(rows):
        normalized = _normalize_row_keys(row)
        question = normalized["question"]
        correct = normalized["correct"]
        incorrects = [
            normalized["incorrect_1"],
            normalized["incorrect_2"],
            normalized["incorrect_3"],
        ]
        if not question or not correct or any(x is None for x in incorrects):
            raise ValueError(
                "GPQA row is missing required fields (question or answers)."
            )

        choices_list = [*incorrects, correct]
        rng.shuffle(choices_list)
        correct_index = choices_list.index(correct)
        choice_map = {
            letter: str(choice) for letter, choice in zip(GPQA_CHOICES, choices_list)
        }

        question_id = normalized["question_id"]
        if question_id is None:
            question_id = idx
        else:
            question_id = int(question_id)

        metadata = {
            key: value
            for key, value in row.items()
            if key
            not in {
                "Question",
                "question",
                "Correct Answer",
                "correct_answer",
                "correct",
                "Incorrect Answer 1",
                "Incorrect Answer 2",
                "Incorrect Answer 3",
                "incorrect_answer_1",
                "incorrect_answer_2",
                "incorrect_answer_3",
                "incorrect_1",
                "incorrect_2",
                "incorrect_3",
            }
        }

        questions.append(
            Question(
                id=question_id,
                original_id=str(question_id),
                question=str(question),
                choices=choice_map,
                answer=GPQA_CHOICES[correct_index],
                metadata=metadata,
            )
        )

    if max_examples is not None:
        questions = questions[:max_examples]

    return BenchmarkDataset(
        name="gpqa",
        questions=questions,
        prompts=GPQA_PROMPTS,
        metadata={
            "dataset_path": dataset_path,
            "subset": subset,
            "split": split,
            "cot_examples_path": GPQA_DEFAULT_COT_EXAMPLES,
        },
    )


def _format_choices(choices: dict[str, str]) -> str:
    """Format answer choices into the GPQA prompt style."""
    lines = []
    for letter in GPQA_CHOICES:
        if letter in choices:
            lines.append(f"({letter}) {choices[letter]}")
    return "\n".join(lines)


def _base_prompt(question: Question) -> str:
    """Create the shared base prompt for GPQA questions."""
    prompt = f"What is the correct answer to this question: {question.question}"
    prompt += f"\n\nChoices:\n{_format_choices(question.choices)}"
    return prompt


def _resolve_resource_path(path: str) -> Path:
    """Resolve resource paths relative to the repo root when needed."""
    candidate = Path(path)
    if candidate.exists():
        return candidate

    repo_root = Path(__file__).resolve().parents[3]
    fallback = repo_root / path
    if fallback.exists():
        return fallback

    return candidate


def _load_cot_examples(path: str) -> list[dict]:
    """Load chain-of-thought example data from a JSON file."""
    resolved_path = _resolve_resource_path(path)
    with resolved_path.open("r") as handle:
        data = json.load(handle)
    return data.get("questions", [])


def _render_cot_examples(examples: list[dict], *, with_explanations: bool) -> str:
    """Render GPQA chain-of-thought examples as prompt text."""
    output = ""
    for item in examples:
        output += f"Question: {item['question']}\nChoices:\n"
        for choice, value in item["choices"].items():
            output += f"({choice}) {value}\n"
        if with_explanations:
            output += f"Let's think step by step: \n{item['explanation']}\n"
        output += f"The correct answer is ({item['correct_answer']})\n"
    return output


def prompt_gpqa(
    dataset: BenchmarkDataset,
    question: Question,
    prompt_type: str = GPQA_DEFAULT_PROMPT_TYPE,
    *,
    cot_examples_path: str | None = None,
) -> str:
    """Build a GPQA prompt in the style of the official baselines."""
    cot_path = cot_examples_path or dataset.metadata.get(
        "cot_examples_path", GPQA_DEFAULT_COT_EXAMPLES
    )

    if prompt_type == "zero_shot":
        prompt = _base_prompt(question)
        prompt += '\n\nFormat your response as follows: "The correct answer is (insert answer here)"'
        return prompt

    if prompt_type == "chain_of_thought":
        examples = _load_cot_examples(cot_path)
        prompt = (
            "Here are some example questions from experts. "
            "An explanation is given before the final answer. "
            "Answer the final question yourself, giving your reasoning beforehand.\n"
        )
        prompt += _render_cot_examples(examples, with_explanations=True)
        prompt += f"Question: {question.question}"
        prompt += f"\nChoices:\n{_format_choices(question.choices)}"
        prompt += (
            "\nGive step by step reasoning before you answer, and when you're ready to answer, "
            'please use the format "The correct answer is (insert answer here)":\n'
        )
        return prompt

    if prompt_type == "5_shot":
        examples = _load_cot_examples(cot_path)
        prompt = (
            "Here are some example questions from experts. "
            "Answer the final question yourself, following the format of the previous questions exactly.\n"
        )
        prompt += _render_cot_examples(examples, with_explanations=False)
        prompt += f"Question: {question.question}"
        prompt += f"\nChoices:\n{_format_choices(question.choices)}"
        prompt += "\nWhen you're ready to answer, please use the format \"The correct answer is (insert answer here)."
        return prompt

    raise ValueError(f"Prompt type '{prompt_type}' is not supported")


def _parse_prediction(text: str) -> ChoiceAnswer:
    """Parse a model response into a GPQA multiple-choice answer."""
    return extract_choice_answer(text, choices=GPQA_CHOICES)


def evaluate_model_on_gpqa(
    *,
    llm_client: LLMClient,
    dataset_path: str = GPQA_DATASET,
    subset: str | None = None,
    split: str = GPQA_DEFAULT_SPLIT,
    seed: int = 0,
    max_examples: int | None = None,
    question_ids: Iterable[int] | None = None,
    prompt_type: str = GPQA_DEFAULT_PROMPT_TYPE,
    cot_examples_path: str = GPQA_DEFAULT_COT_EXAMPLES,
    hf_token: str | None = None,
    hf_token_env: str | None = None,
    request_params: dict[str, object] | None = None,
    show_progress: bool = False,
) -> list[EvalResult]:
    """Evaluate a model on GPQA or GPQA-Diamond and return evaluation results."""
    dataset = load_gpqa_benchmark(
        dataset_path=dataset_path,
        subset=subset,
        split=split,
        seed=seed,
        max_examples=max_examples,
        hf_token=hf_token,
        hf_token_env=hf_token_env,
    )
    dataset = BenchmarkDataset(
        name=dataset.name,
        questions=dataset.questions,
        prompts=dataset.prompts,
        description=dataset.description,
        metadata={**dataset.metadata, "cot_examples_path": cot_examples_path},
        categories=dataset.categories,
        questions_by_category=dataset.questions_by_category,
    )

    def prompt_builder(
        ds: BenchmarkDataset, question: Question, prompt_key: str
    ) -> str:
        return prompt_gpqa(
            ds, question, prompt_key, cot_examples_path=cot_examples_path
        )

    def answer_parser(
        text: str, ds: BenchmarkDataset, question: Question
    ) -> tuple[str | None, dict[str, str]] | str | None:
        parsed = _parse_prediction(text)
        extra = {}
        if parsed.matched_pattern:
            extra["parse_pattern"] = parsed.matched_pattern
        return parsed.choice, extra

    return evaluate_model_on_benchmark(
        dataset,
        llm_client=llm_client,
        prompt_type=prompt_type,
        prompt_builder=prompt_builder,
        answer_parser=answer_parser,
        request_params=request_params,
        question_ids=question_ids,
        show_progress=show_progress,
    )
