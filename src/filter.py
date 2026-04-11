#!/usr/bin/env python3
"""
Filter pipeline for problem datasets.

Filters move FAILING problems out of the directory into failure subdirs.
Passing problems stay in place.

  Step 1: QC filter - move failures to qc_failed/
    python -m src.filter qc <dir> --threshold 0.8

  Step 2: (run indep_solver on <dir> - only passing problems remain)

  Step 3: Solver filter - move failures to solver_failed/
    python -m src.filter solver <dir>

  Step 4: (split remaining files into train/val via io_utils.split_train_val)
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional


# ── Filter functions ──────────────────────────────────────────────


def _get_latest_grading(data: Dict) -> Dict:
    """Get the latest QC grading from quality_gradings. Falls back to legacy 'quality' field."""
    gradings = data.get("quality_gradings", {})
    if gradings:
        for model_id, runs in gradings.items():
            if runs:
                return runs[-1]
    return data.get("quality", {})


def passes_qc(data: Dict, threshold: float = 0.8) -> bool:
    """Check if all QC metrics are above threshold (0-1 scale, converted to 0-100)."""
    quality = _get_latest_grading(data)
    if not quality:
        return False

    threshold_int = int(threshold * 100)

    qc_metrics = [
        "problem_quality",
        "solution_completeness",
        "solution_quality",
    ]

    if "output_seed_correspondence" in quality:
        qc_metrics.append("output_seed_correspondence")

    for metric in qc_metrics:
        score = quality.get(metric)
        if score is None or score < threshold_int:
            return False

    return True


def passes_solver(data: Dict, model: Optional[str] = None) -> bool:
    """Check if problem was independently verified by a solver model."""
    results = data.get("indep_solver_results", {})
    if not results:
        return False

    if model:
        result = results.get(model, {})
        return result.get("passed", False)
    else:
        return any(r.get("passed", False) for r in results.values())


# ── File helpers ──────────────────────────────────────────────────


def gather_problem_files(data_dir: Path) -> List[Path]:
    """Gather problem JSON files, excluding metadata."""
    all_files = sorted(data_dir.glob("*.json"))
    return [f for f in all_files if f.name not in ("assignment.json", "config.json")]


def load_problem(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def move_files(files: List[Path], dest_dir: Path) -> None:
    """Move files to destination directory."""
    dest_dir.mkdir(exist_ok=True)
    for src in files:
        shutil.move(str(src), str(dest_dir / src.name))


# ── Commands ─────────────────────────────────────────────────────


def cmd_qc(args: argparse.Namespace) -> None:
    """Move QC failures to qc_failed/. Passing problems stay in place."""
    base_dir = Path(args.base_dir).resolve()
    problem_files = gather_problem_files(base_dir)
    print(f"Total problem files: {len(problem_files)}")

    failed = []
    passed_count = 0
    for path in problem_files:
        try:
            data = load_problem(path)
            if passes_qc(data, args.threshold):
                passed_count += 1
            else:
                failed.append(path)
        except Exception as e:
            print(f"Warning: {path.name}: {e}")
            failed.append(path)

    print(f"QC filter (>= {args.threshold}): {passed_count} passed, {len(failed)} failed")

    if args.dry_run:
        if failed:
            print(f"[dry-run] Would move {len(failed)} files to qc_failed/")
        return

    if failed:
        move_files(failed, base_dir / "qc_failed")
        print(f"Moved {len(failed)} failures to qc_failed/")


def cmd_solver(args: argparse.Namespace) -> None:
    """Move solver failures to solver_failed/. Passing problems stay in place."""
    base_dir = Path(args.base_dir).resolve()
    problem_files = gather_problem_files(base_dir)
    print(f"Total problem files: {len(problem_files)}")

    failed = []
    passed_count = 0
    for path in problem_files:
        try:
            data = load_problem(path)
            if passes_solver(data, args.solver_model):
                passed_count += 1
            else:
                failed.append(path)
        except Exception as e:
            print(f"Warning: {path.name}: {e}")
            failed.append(path)

    model_str = args.solver_model or "any"
    print(f"Solver filter (model={model_str}): {passed_count} passed, {len(failed)} failed")

    if args.dry_run:
        if failed:
            print(f"[dry-run] Would move {len(failed)} files to solver_failed/")
        return

    if failed:
        move_files(failed, base_dir / "solver_failed")
        print(f"Moved {len(failed)} failures to solver_failed/")


# ── CLI ───────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter problem datasets - move failures out, keep passing in place"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # qc subcommand
    qc_parser = subparsers.add_parser("qc", help="Move QC failures to qc_failed/")
    qc_parser.add_argument("base_dir", type=Path)
    qc_parser.add_argument("--threshold", type=float, default=0.8, help="QC threshold (default: 0.8)")
    qc_parser.add_argument("--dry-run", action="store_true")

    # solver subcommand
    solver_parser = subparsers.add_parser("solver", help="Move solver failures to solver_failed/")
    solver_parser.add_argument("base_dir", type=Path)
    solver_parser.add_argument("--solver-model", type=str, default=None, help="Specific model (default: any)")
    solver_parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()

    if args.command == "qc":
        cmd_qc(args)
    elif args.command == "solver":
        cmd_solver(args)


if __name__ == "__main__":
    main()
