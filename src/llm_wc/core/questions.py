from __future__ import annotations

"""Question schema and defaults for benchmarks."""

from dataclasses import dataclass, field
from typing import Any

DEFAULT_CHOICES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


@dataclass(frozen=True)
class Question:
    """Structured question representation for benchmarking.

    Required fields follow docs/structure.md. Optional fields allow richer
    metadata used by dataset-specific evaluators.
    """

    # Required fields
    id: int  # The question ID unique within the dataset
    original_id: str  # The question ID from the original dataset
    question: str  # The question text
    # The choices for the question. The key is the choice letter, and the value
    # is the choice text. We use strings to index the choices,
    # e.g. "A", "B", "C", "D". The order of the choices is important,
    # and we use the order to map the answer to the correct choice.
    choices: dict[str, str]
    answer: str  # The correct answer to the question

    # Optional fields
    category: str | None = None
    subcategory: str | None = None
    difficulty: str | None = None
    tags: list[str] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
