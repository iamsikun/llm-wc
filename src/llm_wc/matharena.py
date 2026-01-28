from __future__ import annotations

"""MathArena dataset loading and answer parsing helpers."""

import base64
import csv
import json
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterable, Optional

from datasets import load_dataset

# Dataset loading utilities adapted from MathArena's runner.
# These helpers support datasets that live locally (problems/answers/images) or
# on Hugging Face.


def _to_base64_image(image_bytes: bytes | None) -> str | None:
    """Convert raw image bytes into a base64-encoded string.

    Args:
        image_bytes: Raw image bytes from disk or a dataset row.

    Returns:
        str | None: Base64-encoded image string, or None if no bytes were given.
    """
    if image_bytes is None:
        return None
    return base64.b64encode(image_bytes).decode("utf-8")


def load_matharena_competition_problems(
    dataset_path: str,
    *,
    problem_ids: Iterable[int] | None = None,
    final_answer: bool = True,
) -> list[dict]:
    """Load MathArena competition problems from disk or Hugging Face.

    This function is adapted from MathArena's `Runner._load_problems` and keeps
    the same on-disk layout expectations:
        - problems stored under `<dataset_path>/problems/<id>.tex`
        - answers in `<dataset_path>/answers.csv` (final answer)
        - optional images in `<dataset_path>/problems/<id>.png`
        - optional problem metadata in `<dataset_path>/source.csv` and
        `<dataset_path>/problem_types.csv`

    Args:
        dataset_path: Local filesystem path or Hugging Face dataset name.
        problem_ids: Optional list of problem IDs to include.
        final_answer: Whether answers.csv (final answer) should be used. If False,
            grading_scheme.json is expected instead.

    Returns:
        list[dict]: Sorted list of problem dictionaries with keys such as
            "problem_idx", "problem", "answer", and optional "image" metadata.
    """
    normalized_ids = None
    if problem_ids is not None:
        normalized_ids = {int(pid) for pid in problem_ids}

    path = Path(dataset_path)
    if not path.exists():
        # Hugging Face dataset fallback (MathArena publishes many benchmarks here).
        problems = load_dataset(dataset_path, split="train").to_list()
        for problem in problems:
            if "image" in problem and problem["image"] is not None:
                image_bytes = problem["image"].get("bytes")
                problem["image"] = _to_base64_image(image_bytes)
        if normalized_ids is not None:
            problems = [p for p in problems if int(p["problem_idx"]) in normalized_ids]
        return sorted(problems, key=lambda x: x["problem_idx"])

    # Local disk layout mirrors the MathArena repo structure.
    if final_answer:
        answers_path = path / "answers.csv"
    else:
        answers_path = path / "grading_scheme.json"

    source_path = path / "source.csv"
    type_path = path / "problem_types.csv"

    problem_types = None
    if type_path.exists():
        with type_path.open("r") as f:
            problem_types_reader = csv.DictReader(f)
            problem_types = {
                int(row["id"]): row["type"] for row in problem_types_reader
            }
            for problem_id in problem_types:
                problem_types[problem_id] = (
                    problem_types[problem_id]
                    .replace('"', "")
                    .replace("[", "")
                    .replace("]", "")
                    .split(",")
                )

    problems = []
    if final_answer:
        with answers_path.open("r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                id_val = int(row["id"])
                if normalized_ids is not None and id_val not in normalized_ids:
                    continue
                problem_path = path / "problems" / f"{id_val}.tex"
                if problem_path.exists():
                    # Standard text-based math problem.
                    problem_statement = problem_path.read_text()
                    must_have_image = False
                else:
                    # Some MathArena problems are image-only.
                    problem_statement = None
                    must_have_image = True

                image_path = path / "problems" / f"{id_val}.png"
                if image_path.exists():
                    image_b64 = _to_base64_image(image_path.read_bytes())
                else:
                    image_b64 = None
                    if must_have_image:
                        raise FileNotFoundError(
                            f"Problem {id_val} has no text and no image in {dataset_path}"
                        )

                problem_type_val = None
                if problem_types and id_val in problem_types:
                    problem_type_val = problem_types[id_val]

                problems.append(
                    {
                        "problem_idx": id_val,
                        "problem": problem_statement,
                        "image": image_b64,
                        "answer": row["answer"],
                        "problem_type": problem_type_val,
                    }
                )
    else:
        with answers_path.open("r") as f:
            grading_scheme = json.load(f)
        for item in grading_scheme:
            id_val = int(item["id"])
            if normalized_ids is not None and id_val not in normalized_ids:
                continue
            problem_path = path / "problems" / f"{id_val}.tex"
            if problem_path.exists():
                problem_statement = problem_path.read_text()
                must_have_image = False
            else:
                problem_statement = None
                must_have_image = True

            image_path = path / "problems" / f"{id_val}.png"
            if image_path.exists():
                image_b64 = _to_base64_image(image_path.read_bytes())
            else:
                image_b64 = None
                if must_have_image:
                    raise FileNotFoundError(
                        f"Problem {id_val} has no text and no image in {dataset_path}"
                    )

            problem_type_val = None
            if problem_types and id_val in problem_types:
                problem_type_val = problem_types[id_val]

            problems.append(
                {
                    "problem_idx": id_val,
                    "problem": problem_statement,
                    "image": image_b64,
                    "answer": item["scheme"],
                    "problem_type": problem_type_val,
                }
            )

    if source_path.exists():
        with source_path.open("r") as f:
            source_reader = csv.DictReader(f)
            source_map = {int(row["id"]): row["source"] for row in source_reader}
            for problem in problems:
                if problem["problem_idx"] in source_map:
                    problem["source"] = source_map[problem["problem_idx"]]

    return sorted(problems, key=lambda x: x["problem_idx"])


# Math answer parsing utilities adapted from MathArena.
# The original parser handles complex LaTeX parsing with SymPy and a custom
# warning system. This lightweight adaptation focuses on AIME-style integer
# answers while preserving boxed-answer extraction flow.


class WarningLevel(Enum):
    """Warning levels indicating parser confidence."""

    NONE = 0
    MINOR = 1
    MAJOR = 2


@dataclass(frozen=True)
class ParsedAnswer:
    """Parsed answer payload returned by MathArena-style extraction.

    Attributes:
        answer: Parsed answer text (or None when parsing failed).
        warning: WarningLevel describing potential parsing issues.
    """

    answer: Optional[str]
    warning: WarningLevel


def _extract_braced_content(text: str, start_idx: int) -> tuple[Optional[str], int]:
    """Extract a brace-delimited substring starting at the given index.

    Args:
        text: Full text that contains the brace-delimited section.
        start_idx: Index immediately after the opening '{'.

    Returns:
        tuple: (content, end_index) where content excludes the outer braces and
            end_index is the index after the matching closing brace. Returns
            (None, start_idx) if matching braces are not found.
    """
    # Track brace depth to allow nested LaTeX braces.
    depth = 1
    i = start_idx
    while i < len(text):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start_idx:i], i + 1
        i += 1
    return None, start_idx


def _find_boxed_spans(text: str) -> list[tuple[int, int]]:
    """Locate spans for \boxed{...} or \fbox{...} commands.

    Args:
        text: The text to scan.

    Returns:
        list: List of (start, end) spans for each boxed command.
    """
    matches = []
    for match in re.finditer(r"\\(?:boxed|fbox)\{", text):
        content, end_idx = _extract_braced_content(text, match.end())
        if content is not None:
            matches.append((match.start(), end_idx))
    return matches


def _strip_inner_boxes(text: str) -> str:
    """Remove nested boxed commands from a boxed answer snippet.

    Args:
        text: Boxed content that may include nested boxed/fbox commands.

    Returns:
        str: The cleaned content with inner boxed commands removed.
    """
    if "\\boxed" not in text and "\\fbox" not in text:
        return text

    cleaned = text
    while True:
        spans = _find_boxed_spans(cleaned)
        if not spans:
            break
        start, end = spans[-1]
        # Drop the outer command and keep the inner content.
        inner, _ = _extract_braced_content(cleaned, cleaned.find("{", start) + 1)
        if inner is None:
            break
        cleaned = cleaned[:start] + inner + cleaned[end:]
    return cleaned


def find_last_boxed_content(text: str) -> ParsedAnswer:
    """Find the content of the last \boxed or \fbox expression.

    Args:
        text: The text to search.

    Returns:
        ParsedAnswer: Parsed answer and warning level.
    """
    spans = _find_boxed_spans(text)
    if not spans:
        return ParsedAnswer(None, WarningLevel.MAJOR)

    start, end = spans[-1]
    content, _ = _extract_braced_content(text, text.find("{", start) + 1)
    if content is None:
        return ParsedAnswer(None, WarningLevel.MAJOR)
    return ParsedAnswer(_strip_inner_boxes(content).strip(), WarningLevel.NONE)


def extract_last_integer(text: str) -> ParsedAnswer:
    """Extract the last integer token from a string.

    Args:
        text: The text to scan for integers.

    Returns:
        ParsedAnswer: Parsed integer string (or None) plus a warning level.
    """
    matches = list(re.finditer(r"\b\d+\b", text))
    if not matches:
        return ParsedAnswer(None, WarningLevel.MAJOR)
    return ParsedAnswer(matches[-1].group(), WarningLevel.MAJOR)


def extract_answer(text: str, strict_parsing: bool = True) -> ParsedAnswer:
    """Extract an AIME-style answer using MathArena's boxed-answer convention.

    Args:
        text: The model output text to parse.
        strict_parsing: If True, require a boxed answer; if False, fall back to
            the last integer found in the response.

    Returns:
        ParsedAnswer: Parsed answer (string) and warning metadata.
    """
    if text is None or not text.strip():
        return ParsedAnswer(None, WarningLevel.MAJOR)

    boxed = find_last_boxed_content(text)
    if boxed.answer is not None:
        return boxed

    if strict_parsing:
        return boxed

    return extract_last_integer(text)
