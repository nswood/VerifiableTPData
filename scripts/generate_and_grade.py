#!/usr/bin/env python3
"""
Generate model solutions for verifiable problems and grade them via code verification.

Wraps `src.solution_generator.SolutionGenerator` and
`src.solution_grader.CodeVerificationGrader`. Operates on existing problem
JSONs (single file via --problem_path or a directory tree via --problems_dir).
By default, generates solutions and then grades them; pass --grading to skip
generation.

Usage examples:
    # Generate and auto-grade for an entire directory:
    python scripts/generate_and_grade.py \
        --problems_dir synthetic_data/QFT_test/qft_easy_test \
        --num_attempts 5 \
        --model_name gemini-2.5-flash

    # Single problem:
    python scripts/generate_and_grade.py \
        --problem_path synthetic_data/QFT_test/qft_easy_test/p1.json \
        --num_attempts 1 \
        --model_name gpt-5

    # Grading only (skip generation):
    python scripts/generate_and_grade.py \
        --grading \
        --problems_dir synthetic_data/QFT_test/qft_easy_test
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.solution_generator import SolutionGenerator
from src.solution_grader import CodeVerificationGrader


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Generate solutions for verifiable problems and grade them via code verification."
    )

    # Operation mode
    parser.add_argument("--grading", action="store_true", default=False,
                        help="Set to grade solutions only. If not set, generates solutions and then grades them.")

    # Input/output paths
    parser.add_argument("--problems_dir", type=str, default=None,
                        help="Target directory containing problem JSONs (repo-relative). Required unless --problem_path is given.")
    parser.add_argument("--problem_path", type=str, default=None,
                        help="Path to a single problem file. When set, --problems_dir is used only as the root for resolution.")

    # Generation parameters
    parser.add_argument("--model_name", type=str, default="gemini-2.5-flash", help="Model name to use.")
    parser.add_argument("--num_attempts", type=int, default=1, help="Number of attempts to generate solutions.")
    parser.add_argument("--prompt_type", type=str, default="CoT", choices=["standard", "CoT"],
                        help="Type of prompt to use.")
    parser.add_argument("--temperature", type=float, default=None,
                        help="Temperature for generation. If not provided, uses each model's default temperature.")
    parser.add_argument("--multi_gpu", type=int, default=1, help="Number of GPUs to use for local models.")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.95,
                        help="GPU memory utilization fraction (0.0-1.0) for vLLM. Default: 0.95")
    parser.add_argument("--model_alias", type=str, default=None,
                        help="Custom model name for results (e.g., 'oss-20b-100-step'). If not set, uses model_name.")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Global batch size for solution generation (shared with vLLM batching). Defaults to large value for API models, dynamic for vLLM.")
    parser.add_argument("--requests_per_minute", type=int, default=1000,
                        help="Rate limit for API requests per minute.")
    parser.add_argument("--max_concurrent_requests", type=int, default=500,
                        help="Maximum number of concurrent API requests.")
    parser.add_argument("--reasoning_level", type=str, default=None, choices=["low", "medium", "high"],
                        help="Reasoning level for oss models (low/medium/high). Prepends 'Reasoning: <level>' to system prompt.")

    # Behavior flags
    parser.add_argument("--overwrite_attempts", action="store_true", default=False,
                        help="Set to overwrite existing attempts instead of appending.")
    parser.add_argument("--regrade_all", action="store_true", default=False,
                        help="Regrade all attempts, even if already graded (default: skip already graded).")

    # Grading parameters
    parser.add_argument("--quiet", action="store_true", default=False,
                        help="Suppress debug output during grading.")

    # Configuration
    parser.add_argument("--config_path", type=str, default=None,
                        help="Path to custom config file. If unset, the generator/grader use their packaged defaults under configs/.")

    # Reporting
    parser.add_argument("--latex_report", action=argparse.BooleanOptionalAction, default=True,
                        help="Run LaTeX analysis report (PDF) after grading all problems. Use --no-latex_report to skip.")

    args = parser.parse_args()

    if args.problems_dir is None and args.problem_path is None:
        parser.error("Either --problems_dir or --problem_path must be provided.")

    # When only --problem_path is given, derive --problems_dir from its parent
    # so BaseProblemProcessor can build absolute paths. We make it relative to
    # the repo root if possible, otherwise pass through absolute.
    if args.problems_dir is None:
        problem_path = Path(args.problem_path).resolve()
        parent = problem_path.parent
        try:
            args.problems_dir = str(parent.relative_to(REPO_ROOT))
        except ValueError:
            args.problems_dir = str(parent)

    return args


def create_grader(args):
    grader_kwargs = {
        "problems_dir": args.problems_dir,
        "quiet": args.quiet,
    }
    if args.config_path:
        grader_kwargs["config_path"] = args.config_path
    return CodeVerificationGrader(**grader_kwargs)


def create_generator(args):
    generator_kwargs = {
        "model_name": args.model_name,
        "prompt_type": args.prompt_type,
        "multi_gpu": args.multi_gpu,
        "problems_dir": args.problems_dir,
        "overwrite_attempts": args.overwrite_attempts,
        "batch_size": args.batch_size,
        "requests_per_minute": args.requests_per_minute,
        "max_concurrent_requests": args.max_concurrent_requests,
        "gpu_memory_utilization": args.gpu_memory_utilization,
    }
    if args.config_path:
        generator_kwargs["config_path"] = args.config_path
    if args.temperature is not None:
        generator_kwargs["temperature"] = args.temperature
    if args.model_alias:
        generator_kwargs["model_alias"] = args.model_alias
    if args.reasoning_level:
        generator_kwargs["reasoning_level"] = args.reasoning_level
        if "oss" in args.model_name:
            base_alias = args.model_alias if args.model_alias else args.model_name
            generator_kwargs["model_alias"] = f"{base_alias}-{args.reasoning_level}"
    return SolutionGenerator(**generator_kwargs)


def run_latex_report(args):
    """Run latex generation for analysis report, if a generator script exists in this repo."""
    logger = logging.getLogger(__name__)

    candidates = [
        REPO_ROOT / "scripts" / "generate_latex.py",
        REPO_ROOT / "main" / "generate_latex.py",
    ]
    script = next((p for p in candidates if p.exists()), None)
    if script is None:
        logger.warning(
            "Latex report requested, but no generate_latex.py was found at %s. Skipping.",
            " or ".join(str(p) for p in candidates),
        )
        return

    cmd = [
        sys.executable,
        str(script),
        "--analysis_only",
        "--pdf",
        "--problems_dir", args.problems_dir,
    ]
    if args.config_path:
        cmd.extend(["--config_path", args.config_path])

    logger.info("Generating analysis report via %s ...", script)
    try:
        subprocess.run(cmd, check=True)
        logger.info("Analysis report generated successfully.")
    except subprocess.CalledProcessError as e:
        logger.error("Failed to generate analysis report: %s", e)


def run_grading(args):
    logger = logging.getLogger(__name__)
    grader = create_grader(args)
    logger.info("%s", grader)

    if args.problem_path is not None:
        grader._grade_problem(
            problem_path=args.problem_path,
            regrade_all=args.regrade_all,
        )
    else:
        logger.info("Grading all problems")
        grader.grade_all_problems(regrade_all=args.regrade_all)
        if args.latex_report:
            run_latex_report(args)


def run_generation(args):
    generator = create_generator(args)
    try:
        if args.problem_path is not None:
            generator.generate_solutions_for_problem(
                problem_path=args.problem_path,
                num_attempts=args.num_attempts,
            )
        else:
            generator.generate_solutions_for_all_problems(num_attempts=args.num_attempts)
    finally:
        generator._terminate_subprocesses()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    args = parse_arguments()
    args.problem_path = Path(args.problem_path) if args.problem_path else None

    if args.grading:
        run_grading(args)
    else:
        run_generation(args)
        # Auto-grade after generation
        run_grading(args)


if __name__ == "__main__":
    main()
