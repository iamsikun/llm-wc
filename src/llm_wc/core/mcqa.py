from __future__ import annotations

"""Multiple-choice QA utilities shared across benchmarks."""

import re
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class ChoiceAnswer:
    """Container for extracted multiple-choice answers.

    Attributes:
        choice: Extracted choice letter, or None if parsing failed.
        matched_pattern: Regex pattern that matched (useful for debugging).
    """

    choice: str | None
    matched_pattern: str | None


def normalize_choices(choices: Iterable[str]) -> list[str]:
    """Normalize a collection of choice labels into a sorted list.

    Args:
        choices: Iterable of choice labels (e.g., ["A", "B", "C", "D"]).

    Returns:
        list[str]: Sorted, uppercased choice labels.
    """
    return sorted(
        {str(choice).strip().upper() for choice in choices if str(choice).strip()}
    )


def extract_choice_answer(
    text: str,
    *,
    choices: Iterable[str],
    patterns: list[str] | None = None,
) -> ChoiceAnswer:
    """Extract a multiple-choice answer letter from model output.

    Args:
        text: The model output to parse.
        choices: Valid choice labels (e.g., A-D).
        patterns: Optional regex patterns to try before fallback extraction.

    Returns:
        ChoiceAnswer: Parsed choice (or None) and the matching pattern.
    """
    if text is None:
        return ChoiceAnswer(None, None)

    valid_choices = normalize_choices(choices)
    if not valid_choices:
        return ChoiceAnswer(None, None)

    pattern_list = patterns or [
        r"answer is \(?([A-Z])\)?",
        r"correct answer is \(?([A-Z])\)?",
        r"Answer: \(?([A-Z])\)?",
        r"answer: \(?([A-Z])\)?",
        r"answer \(?([A-Z])\)?",
        r"\(([A-Z])\)",
    ]

    for pattern in pattern_list:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            choice = match.group(1).upper()
            if choice in valid_choices:
                return ChoiceAnswer(choice, pattern)

    # Fallback: pick the last standalone choice letter in the response.
    choice_regex = (
        r"\b("
        + "|".join(map(re.escape, valid_choices))
        + r")\b(?!.*\b("
        + "|".join(map(re.escape, valid_choices))
        + r")\b)"
    )
    match = re.search(choice_regex, text, re.IGNORECASE | re.DOTALL)
    if match:
        return ChoiceAnswer(match.group(1).upper(), "fallback_last_choice")

    return ChoiceAnswer(None, None)
