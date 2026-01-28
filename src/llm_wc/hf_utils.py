from __future__ import annotations

"""Hugging Face dataset helpers for gated or authenticated access."""

from typing import Iterable

from datasets import load_dataset

DEFAULT_HF_TOKEN_ENVS = ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN")


def resolve_hf_token(
    hf_token: str | None,
    hf_token_env: str | None,
    *,
    default_envs: Iterable[str] = DEFAULT_HF_TOKEN_ENVS,
    environ: dict[str, str] | None = None,
) -> str | None:
    """Resolve a Hugging Face token from explicit input or env vars.

    Args:
        hf_token: Explicit token override.
        hf_token_env: Environment variable name to read first.
        default_envs: Fallback env var names to check in order.
        environ: Optional environment mapping (defaults to os.environ).

    Returns:
        str | None: Resolved token or None if no token is found.
    """
    if hf_token:
        return hf_token

    env = environ
    if env is None:
        import os

        env = os.environ

    if hf_token_env:
        value = env.get(hf_token_env)
        if value:
            return value

    for env_name in default_envs:
        value = env.get(env_name)
        if value:
            return value

    return None


def load_dataset_with_token(*args, hf_token: str | None = None, **kwargs):
    """Compatibility wrapper for datasets.load_dataset auth kwarg.

    Args:
        *args: Positional args for datasets.load_dataset.
        hf_token: Optional token for gated dataset access.
        **kwargs: Keyword args for datasets.load_dataset.

    Returns:
        Dataset: Loaded datasets object.
    """
    if hf_token is None:
        return load_dataset(*args, **kwargs)
    try:
        return load_dataset(*args, token=hf_token, **kwargs)
    except TypeError:
        # Older versions of datasets use `use_auth_token`.
        return load_dataset(*args, use_auth_token=hf_token, **kwargs)


def is_gated_dataset_error(exc: Exception) -> bool:
    """Heuristic check for gated dataset errors.

    Args:
        exc: Exception raised by datasets.load_dataset.

    Returns:
        bool: True if the error looks like a gated dataset auth failure.
    """
    msg = str(exc).lower()
    return any(
        token in msg
        for token in (
            "gated dataset",
            "must be authenticated",
            "access to this dataset",
            "permission",
        )
    )


def load_hf_dataset_rows(
    dataset_path: str,
    *,
    subset: str | None = None,
    split: str = "train",
    hf_token: str | None = None,
    hf_token_env: str | None = None,
) -> list[dict]:
    """Load dataset rows from Hugging Face with optional authentication.

    Args:
        dataset_path: Hugging Face dataset identifier.
        subset: Optional dataset configuration (e.g., "diamond").
        split: Dataset split name to load.
        hf_token: Optional Hugging Face token to access gated datasets.
        hf_token_env: Optional env var name to read the Hugging Face token from.

    Returns:
        list[dict]: List of dataset rows.
    """
    resolved_token = resolve_hf_token(hf_token, hf_token_env)
    if subset:
        dataset = load_dataset_with_token(
            dataset_path, subset, split=split, hf_token=resolved_token
        )
    else:
        dataset = load_dataset_with_token(
            dataset_path, split=split, hf_token=resolved_token
        )
    return dataset.to_list()
