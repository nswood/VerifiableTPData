"""
Independent Solver - Verifies problem solvability with pass@k evaluation.

This module provides an independent solver verification pipeline that:
1. Takes QC-passed problems and has LLM models attempt to solve them
2. Grades each attempt immediately after generation
3. Stops early if an attempt passes (no wasted API calls)
4. Reports pass@k metrics

Usage:
    python -m src.indep_solver \
        --problems_dir synthetic_data/QFT/dataset \
        --model "gemini-2.5-flash" \
        --max_k 3 \
        --dry-run
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Independent solver verification with pass@k evaluation"
    )
    parser.add_argument(
        "--problems_dir",
        type=str,
        required=True,
        help="Directory containing problem JSON files"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-2.5-flash",
        help="Model to use for solving (default: gemini-2.5-flash)"
    )
    parser.add_argument(
        "--max_k",
        type=int,
        default=3,
        help="Maximum attempts for pass@k (default: 3)"
    )
    parser.add_argument(
        "--prompt_type",
        type=str,
        default="CoT",
        choices=["CoT", "standard"],
        help="Prompt type (default: CoT)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making API calls"
    )
    parser.add_argument(
        "--regrade",
        action="store_true",
        help="Re-run grading on existing attempts"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output"
    )
    return parser.parse_args()


def find_problem_files(problems_dir: Path) -> List[Path]:
    """Find all problem JSON files in directory."""
    return sorted(problems_dir.glob("p*.json"))


def load_problem(path: Path) -> Dict:
    """Load problem JSON file."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_problem(problem: Dict, path: Path) -> None:
    """Save problem JSON file."""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(problem, f, indent=2, ensure_ascii=False)


def has_existing_result(problem: Dict, model: str) -> Optional[Dict]:
    """Check if problem already has indep_solver result for this model."""
    results = problem.get("indep_solver_results", {})
    return results.get(model)


def solve_and_grade_single_attempt(
    problem_path: Path,
    model: str,
    prompt_type: str,
) -> Dict:
    """
    Generate a single solution attempt, grade it, and save results to the problem JSON.

    The generator saves the attempt to model_solutions in the JSON.
    The grader adds verifier_result to the attempt.

    Returns:
        Dict with keys: verified (0/1), error (optional)
    """
    # Import here to avoid loading heavy deps on dry-run
    from .solution_generator import SolutionGenerator
    from .solution_grader import CodeVerificationGrader

    # Generate single attempt - this saves model_solutions to the JSON
    generator = SolutionGenerator(
        model_name=model,
        prompt_type=prompt_type,
        problems_dir=str(problem_path.parent),
        quiet=True,
    )
    attempts = generator.generate_solution(problem_path, num_attempts=1)

    if not attempts:
        return {"verified": 0, "error": "No solution generated"}

    # Reload problem from disk (generator may have saved model_solutions)
    problem = load_problem(problem_path)
    attempt = attempts[0]

    # Grade the attempt
    grader = CodeVerificationGrader(
        problems_dir=str(problem_path.parent),
        quiet=True,
    )
    verifier_result = grader._grade_solution(attempt, problem, model)

    # Save verifier_result onto the attempt in model_solutions
    for ms in problem.get("model_solutions", []):
        if ms.get("model") == model:
            for a in ms.get("attempts", []):
                # Match by timestamp
                if a.get("timestamp") == attempt.get("timestamp"):
                    a["verifier_result"] = verifier_result
                    break
            break

    save_problem(problem, problem_path)

    return {"verified": verifier_result.get("verified", 0)}


def run_indep_solver(
    problems_dir: Path,
    model: str,
    max_k: int,
    prompt_type: str,
    dry_run: bool = False,
    regrade: bool = False,
    quiet: bool = False,
) -> Dict[str, Any]:
    """
    Run independent solver verification on all problems.

    Args:
        problems_dir: Directory containing problem JSON files
        model: Model to use for solving
        max_k: Maximum attempts for pass@k
        prompt_type: Prompt type (CoT or standard)
        dry_run: If True, don't make API calls
        regrade: If True, re-run grading on existing attempts
        quiet: If True, suppress progress output

    Returns:
        Summary statistics dict
    """
    problem_files = find_problem_files(problems_dir)

    if not problem_files:
        print(f"No problem files found in {problems_dir}")
        return {}

    if dry_run:
        print(f"[dry-run] Would process {len(problem_files)} problems")
        print(f"[dry-run] Model: {model}")
        print(f"[dry-run] Max attempts (k): {max_k}")
        print(f"[dry-run] Prompt type: {prompt_type}")

        # Show sample of problems
        print(f"\n[dry-run] Problems to process:")
        for i, pf in enumerate(problem_files[:5]):
            problem = load_problem(pf)
            pid = problem.get("problem_id", pf.stem)
            existing = has_existing_result(problem, model)
            status = "has result" if existing else "needs solving"
            print(f"[dry-run]   {i+1}. {pid}: {status}")
        if len(problem_files) > 5:
            print(f"[dry-run]   ... and {len(problem_files) - 5} more")

        return {"dry_run": True, "problem_count": len(problem_files)}

    # Pre-scan to determine which problems need solving
    to_solve = []
    skipped_count = 0
    for problem_path in problem_files:
        problem = load_problem(problem_path)
        existing = has_existing_result(problem, model)
        if existing and not regrade:
            if existing.get("passed"):
                skipped_count += 1
                continue
            prev_attempts = existing.get("attempts_needed", 0)
            if prev_attempts >= max_k:
                skipped_count += 1
                continue
        to_solve.append(problem_path)

    print(f"Total problems: {len(problem_files)}, need solving: {len(to_solve)}, skipped: {skipped_count}")

    # Stats tracking
    stats = {
        "total_problems": len(problem_files),
        "passed": 0,
        "failed": 0,
        "skipped": skipped_count,
        "total_attempts": 0,
        "attempts_saved_by_early_stop": 0,
        "pass_at_k": {k: 0 for k in range(1, max_k + 1)},
    }

    if not to_solve:
        print("Nothing to solve.")
        return stats

    pbar = tqdm(to_solve, desc="Solving problems", disable=quiet)

    for problem_path in pbar:
        problem = load_problem(problem_path)
        pid = problem.get("problem_id", problem_path.stem)

        # Determine starting attempt (resume support)
        existing = has_existing_result(problem, model)
        if existing and not regrade and not existing.get("passed"):
            start_attempt = existing.get("attempts_needed", 0) + 1
        else:
            start_attempt = 1

        # Try attempts with early stopping
        passed = False
        attempts_used = 0

        for attempt_num in range(start_attempt, max_k + 1):
            attempts_used = attempt_num
            stats["total_attempts"] += 1

            pbar.set_postfix({"status": f"{pid}: attempt {attempt_num}/{max_k}"})

            try:
                result = solve_and_grade_single_attempt(
                    problem_path=problem_path,
                    model=model,
                    prompt_type=prompt_type,
                )

                if result["verified"] == 1:
                    passed = True
                    for k in range(attempts_used, max_k + 1):
                        stats["pass_at_k"][k] += 1
                    stats["attempts_saved_by_early_stop"] += (max_k - attempt_num)
                    break

            except Exception as e:
                if not quiet:
                    print(f"\nError solving {pid} attempt {attempt_num}: {e}")
                continue

        if passed:
            stats["passed"] += 1
        else:
            stats["failed"] += 1

        # Save pass@k summary to problem JSON
        problem = load_problem(problem_path)  # reload to pick up model_solutions
        if "indep_solver_results" not in problem:
            problem["indep_solver_results"] = {}

        problem["indep_solver_results"][model] = {
            "passed": passed,
            "attempts_needed": attempts_used if passed else max_k,
            "max_k": max_k,
            "prompt_type": prompt_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        save_problem(problem, problem_path)

    pbar.close()

    # Clean up .lock files left by the solution generator
    for lock_file in problems_dir.glob("*.lock"):
        lock_file.unlink()

    # Print summary
    print("\n" + "=" * 60)
    print("INDEPENDENT SOLVER RESULTS")
    print("=" * 60)
    print(f"Model: {model}")
    print(f"Total problems: {stats['total_problems']}")
    print(f"Skipped (existing): {stats['skipped']}")
    print(f"Passed: {stats['passed']}")
    print(f"Failed: {stats['failed']}")
    print()

    evaluated = stats['passed'] + stats['failed']
    if evaluated > 0:
        print("Pass@k Results:")
        for k in range(1, max_k + 1):
            count = stats["pass_at_k"][k]
            pct = 100.0 * count / evaluated
            print(f"  Pass@{k}: {count}/{evaluated} ({pct:.1f}%)")

        print()
        print(f"Total API calls: {stats['total_attempts']}")
        print(f"API calls saved by early stopping: {stats['attempts_saved_by_early_stop']}")

    print("=" * 60)

    return stats


def main():
    args = parse_args()

    problems_dir = Path(args.problems_dir)
    if not problems_dir.exists():
        print(f"Error: Problems directory not found: {problems_dir}")
        return

    run_indep_solver(
        problems_dir=problems_dir,
        model=args.model,
        max_k=args.max_k,
        prompt_type=args.prompt_type,
        dry_run=args.dry_run,
        regrade=args.regrade,
        quiet=args.quiet,
    )


if __name__ == "__main__":
    main()
