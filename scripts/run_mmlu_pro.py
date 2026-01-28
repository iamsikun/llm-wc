#!/usr/bin/env python3
"""Run the MMLU-Pro benchmark evaluation from a YAML configuration file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from dataclasses import asdict

from llm_wc.bench_cli import (
    build_output_dir,
    filter_available_models,
    parse_list,
    parse_model_filter,
    sanitize_name,
    write_run_metadata,
)
from llm_wc.bench_utils import (
    apply_overrides,
    build_logger,
    load_yaml_config,
    parse_run_config,
    resolve_api_key,
    RunConfig,
)
from llm_wc.client import ClientConfig, build_client
from llm_wc.core.eval import compute_accuracy
from llm_wc.mmlu_pro import evaluate_model_on_mmlu_pro


def main() -> int:
    """CLI entry point for running the MMLU-Pro evaluation."""
    parser = argparse.ArgumentParser(description="Run MMLU-Pro evaluation from config")
    parser.add_argument(
        "--config",
        default="config/mmlu_pro.yaml",
        help="Path to YAML config (default: config/mmlu_pro.yaml)",
    )
    parser.add_argument("--output-dir", help="Override benchmark.output_dir")
    parser.add_argument("--seed", type=int, help="Override benchmark.seed")
    parser.add_argument(
        "--limit-subjects",
        help="Comma-separated subject list (use 'none' for all)",
    )
    parser.add_argument(
        "--limit-per-subject",
        type=int,
        help="Override benchmark.limit_per_subject",
    )
    parser.add_argument(
        "--models",
        help="Comma-separated model names to run (filters config models)",
    )
    parser.add_argument(
        "--api-base",
        help="Override api_base for all models",
    )
    parser.add_argument(
        "--api-key",
        help="Override api_key for all models",
    )
    parser.add_argument(
        "--api-key-env",
        help="Override api_key_env for all models",
    )
    parser.add_argument("--temperature", type=float, help="Override temperature")
    parser.add_argument("--top-p", type=float, help="Override top_p")
    parser.add_argument("--max-tokens", type=int, help="Override max_tokens")
    parser.add_argument(
        "--frequency-penalty", type=float, help="Override frequency_penalty"
    )
    parser.add_argument(
        "--presence-penalty", type=float, help="Override presence_penalty"
    )
    parser.add_argument("--request-seed", type=int, help="Override request seed")
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (default: INFO)",
    )
    args = parser.parse_args()

    logger = build_logger("llm_wc.bench.mmlu_pro", level=args.log_level)

    config_path = Path(args.config)
    cfg_data = load_yaml_config(config_path)
    run_config = parse_run_config(cfg_data)

    request_override = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "frequency_penalty": args.frequency_penalty,
        "presence_penalty": args.presence_penalty,
        "seed": args.request_seed,
    }

    limit_subjects = parse_list(args.limit_subjects)
    run_config = apply_overrides(
        run_config,
        output_dir=args.output_dir,
        seed=args.seed,
        limit_subjects=limit_subjects,
        limit_subjects_set=args.limit_subjects is not None,
        limit_per_subject=args.limit_per_subject,
        model_names=parse_model_filter(args.models),
        api_base=args.api_base,
        api_key=args.api_key,
        api_key_env=args.api_key_env,
        request_overrides=request_override,
    )

    benchmark = run_config.benchmark
    output_dir = build_output_dir(config_path, args.output_dir)

    if not run_config.models:
        raise ValueError("No models to evaluate after applying filters")

    logger.info(
        "Starting %s evaluation with %d model(s)",
        benchmark.name,
        len(run_config.models),
    )

    run_config = filter_available_models(
        run_config, logger, resolve_api_key=resolve_api_key
    )
    if not run_config.models:
        logger.warning("No available models to run after availability check")
        return 1

    write_run_metadata(output_dir, run_config)

    for model_cfg in run_config.models:
        model_name = model_cfg.name
        api_key = resolve_api_key(model_cfg)
        request_params = dict(model_cfg.request_overrides)

        model_dir_name = model_cfg.output_name or sanitize_name(model_name)
        model_output_dir = output_dir / "models" / model_dir_name
        model_output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Evaluating model '%s' on %s", model_name, benchmark.name)
        llm_client = build_client(
            ClientConfig(
                provider="openai",
                model=model_name,
                api_base=model_cfg.api_base,
                api_key=api_key,
            )
        )
        results = evaluate_model_on_mmlu_pro(
            llm_client=llm_client,
            subjects=benchmark.limit_subjects,
            limit_per_subject=benchmark.limit_per_subject,
            prompt_type="cot",
            request_params=request_params,
            show_progress=True,
        )

        (model_output_dir / "results.json").write_text(
            json.dumps([asdict(result) for result in results], indent=2)
        )

        summary = compute_accuracy(results)
        summary.update(
            {
                "model": model_name,
                "api_base": model_cfg.api_base,
                "request_params": request_params,
                "seed": benchmark.seed,
            }
        )
        (model_output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        logger.info(
            "Completed model '%s': accuracy=%.4f",
            model_name,
            summary["accuracy"],
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
