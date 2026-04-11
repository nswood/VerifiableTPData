#!/usr/bin/env python3
"""
Run the full data pipeline: generate -> qc -> filter qc -> indep solver -> filter solver.

Auto-detects whether the config is for synthetic or adapted generation based on
the presence of 'seed_data_path' in the YAML.

Usage:
    python scripts/run_pipeline.py --config configs/test/easy_test.yaml
    python scripts/run_pipeline.py --config configs/test/adapted_mit_ocw_test.yaml --solver-model gemini-2.5-flash --max-k 3
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_output_dir(config: dict) -> Path:
    """Derive the output directory from a generator config."""
    topic_name = config.get("topic_name", "")
    dataset_name = config["dataset_name"]

    # Adapted generator can override data_type; synthetic always uses synthetic_data
    if "seed_data_path" in config:
        data_type = config.get("data_type", "semi_synthetic_data")
    else:
        data_type = "synthetic_data"

    return Path(data_type) / topic_name / dataset_name


def run_step(label: str, cmd: list[str]) -> None:
    print(f"\n{'=' * 60}")
    print(f"STAGE: {label}")
    print(f"CMD: {' '.join(cmd)}")
    print("=" * 60)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\nStage '{label}' failed with exit code {result.returncode}")
        sys.exit(result.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full data pipeline")
    parser.add_argument("--config", type=Path, required=True, help="Generator config YAML")
    parser.add_argument("--qc-threshold", type=float, default=0.8, help="QC pass threshold (default: 0.8)")
    parser.add_argument("--solver-model", type=str, default="gemini-2.5-flash", help="Independent solver model")
    parser.add_argument("--max-k", type=int, default=3, help="Pass@k attempts for solver (default: 3)")
    parser.add_argument("--skip-generate", action="store_true", help="Skip the generation stage")
    parser.add_argument("--skip-qc", action="store_true", help="Skip QC + qc filter stages")
    parser.add_argument("--skip-solver", action="store_true", help="Skip indep solver + solver filter stages")
    args = parser.parse_args()

    if not args.config.exists():
        print(f"Config not found: {args.config}")
        sys.exit(1)

    config = load_config(args.config)
    out_dir = get_output_dir(config)

    is_adapted = "seed_data_path" in config
    generator_module = "src.adapted_generator" if is_adapted else "src.generator"

    print(f"Pipeline config: {args.config}")
    print(f"Generator: {generator_module}")
    print(f"Output dir: {out_dir}")

    # Stage 1: Generate
    if not args.skip_generate:
        run_step("1/5 Generate", ["python", "-m", generator_module, "--config", str(args.config)])

    # Stage 2: Quality check
    if not args.skip_qc:
        run_step(
            "2/5 Quality Check",
            ["python", "-m", "src.quality_check_problems", "--base_dir", str(out_dir)],
        )

        # Stage 3: Filter by QC
        run_step(
            "3/5 Filter QC",
            ["python", "-m", "src.filter", "qc", str(out_dir), "--threshold", str(args.qc_threshold)],
        )

    # Stage 4: Independent solver
    if not args.skip_solver:
        run_step(
            "4/5 Independent Solver",
            [
                "python", "-m", "src.indep_solver",
                "--problems_dir", str(out_dir),
                "--model", args.solver_model,
                "--max_k", str(args.max_k),
            ],
        )

        # Stage 5: Filter by solver
        run_step(
            "5/5 Filter Solver",
            ["python", "-m", "src.filter", "solver", str(out_dir)],
        )

    print(f"\n{'=' * 60}")
    print("PIPELINE COMPLETE")
    print(f"{'=' * 60}")
    print(f"Verified problems remain in: {out_dir}/")
    print(f"Failed problems are in: {out_dir}/qc_failed/ and {out_dir}/solver_failed/")


if __name__ == "__main__":
    main()
