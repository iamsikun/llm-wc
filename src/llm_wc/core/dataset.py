from __future__ import annotations

"""Dataset representation for benchmarking."""

import warnings
from dataclasses import dataclass, field, replace
from typing import Any

from llm_wc.core.questions import Question


@dataclass(frozen=True)
class BenchmarkDataset:
    """Dataset container used by the evaluation layer.

    Required fields follow docs/structure.md. Optional fields support
    dataset-specific metadata and categorization.
    """

    # Required fields
    name: str  # The name of the dataset
    questions: list[Question]  # The list of questions in the dataset
    prompts: dict[str, str]  # The prompts for the dataset

    # Optional fields
    description: str | None = None  # The description of the dataset
    metadata: dict[str, Any] = field(default_factory=dict)
    categories: list[str] | None = None  # The categories of the dataset
    questions_by_category: dict[str, list[Question]] | None = None

    def with_questions_by_category(self) -> BenchmarkDataset:
        """Return a dataset with questions grouped by category."""
        if self.questions_by_category is not None:
            return self
        grouped: dict[str, list[Question]] = {}
        for question in self.questions:
            if question.category is None:
                continue
            grouped.setdefault(question.category, []).append(question)
        categories = self.categories
        if categories is None and grouped:
            categories = list(grouped.keys())
        return replace(
            self,
            categories=categories,
            questions_by_category=grouped or None,
        )

    def get_questions_by_category(self, category: str) -> list[Question]:
        """Get the questions by category.

        Args:
            category: The category to get the questions from.

        Returns:
            A list of questions in the category.
        """
        dataset = self.with_questions_by_category()
        if dataset.questions_by_category:
            return dataset.questions_by_category.get(category, [])
        if dataset.categories is None:
            warnings.warn(
                f"Categories information is not available in dataset {self.name}"
            )
            return dataset.questions
        if category not in dataset.categories:
            warnings.warn(f"Category {category} not found in dataset {self.name}")
            return []
        return [
            question for question in dataset.questions if question.category == category
        ]
