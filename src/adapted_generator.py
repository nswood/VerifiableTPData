from __future__ import annotations

import argparse
import hashlib
import json
import re
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date
from multiprocessing import Manager, Value
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import yaml
from tqdm import tqdm

from . import io_utils
from . import llm_client
from . import parser
from . import seed_loader
from . import generate_test_cases

try:
    # genai.py contains all Gemini/GenAI configuration and functions
    from genai import get_api_key  # type: ignore
except Exception:  # pragma: no cover - robust import fallback
    def get_api_key() -> str:  # type: ignore
        return ""


def _wrap_code_section(code: str) -> str:
    """
    Wrap code content in LaTeX python environment.
    Strips any existing markdown code blocks (```python ... ```) before wrapping.
    """
    # Strip markdown code blocks if present
    code = re.sub(r'```python\s*\n?', '', code)
    code = re.sub(r'```\s*$', '', code, flags=re.MULTILINE)
    code = code.strip()
    
    # Wrap in LaTeX python environment
    return f"\\begin{{python}}\n{code}\n\\end{{python}}"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate semi-synthetic problems from seed data")
    p.add_argument("--config", type=str, required=True, help="Path to YAML config")
    p.add_argument("--dry-run", action="store_true", help="List seed pairs only; no writes")
    return p.parse_args()


def load_config(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def compute_pair_hash(arxiv_id: str, problem: str, solution: str) -> str:
    """
    Compute a hash identifier for a seed pair to track which pairs have been processed.
    
    Args:
        arxiv_id: Arxiv ID
        problem: Problem text
        solution: Solution text
        
    Returns:
        Hash string identifier
    """
    # Create a unique identifier from arxiv_id and problem text
    # Using problem text is sufficient since it uniquely identifies the pair
    content = f"{arxiv_id}:{problem}"
    return hashlib.md5(content.encode('utf-8')).hexdigest()


def load_processed_pair_hashes(out_dir: Path) -> Set[str]:
    """
    Load hashes of already processed seed pairs from existing problem files.
    
    Args:
        out_dir: Output directory containing problem JSON files
        
    Returns:
        Set of pair hash strings
    """
    processed_hashes: Set[str] = set()
    
    if not out_dir.exists():
        return processed_hashes

    # Scan top-level AND subdirs (qc_failed/, solver_failed/, etc.)
    # so resume works after filter steps move files out
    for json_file in out_dir.rglob("p*.json"):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                problem_data = json.load(f)
            
            # Check for stored pair hash in metadata
            metadata = problem_data.get("problem_metadata", {})
            pair_hash = metadata.get("Seed Pair Hash")
            if pair_hash:
                processed_hashes.add(pair_hash)
        except (json.JSONDecodeError, IOError):
            continue
    
    return processed_hashes


def process_seed_pair(
    seed_problem: str,
    seed_solution: str,
    arxiv_id: str,
    problem_idx: int,
    system_prompt_template: str,
    out_dir: Path,
    api_key: str,
    model: str,
    topic_name: str,
    verbose: bool,
    parse_code_subsections: bool = False,
    seed_type: str = "matched",  # NEW
    seed_data_dict: Optional[Dict[str, Any]] = None,  # NEW
) -> Optional[Dict[str, Any]]:
    """
    Process a single seed problem-solution pair through LLM and save the result.
    
    Args:
        seed_problem: Original problem text
        seed_solution: Original solution text
        arxiv_id: Arxiv ID for this seed pair
        problem_idx: Problem index for output filename
        system_prompt_template: System prompt template
        out_dir: Output directory
        api_key: API key
        model: Model name
    topic_name: Topic name (e.g., "QFT")
    verbose: Whether to save prompts
    parse_code_subsections: If True, parser expects Code section with \\subsection{Task N} markers.
                           If False (default), parser extracts entire Code section as single block.
    
    Returns:
        Problem data dictionary if successful, None otherwise
    """
    try:
        # Inject task instructions placeholder (will be replaced if needed)
        # For semi-synthetic, we let the LLM determine tasks from the seed problem
        system_prompt = system_prompt_template#.replace(
        #     "{{TASK_INSTRUCTIONS}}",
        #     "Extract or create tasks from the original problem. Modify the problem minimally such that each task fits the requirements listed below. You may remove tasks if they are not possible to reformat. The aim to preserve the original problem as much as possible."
        # )
        
        # Replace level placeholder (semi-synthetic doesn't have level info, use generic term)
        system_prompt = system_prompt.replace("{{INSERT LEVEL}}", "graduate")
        
        # Generate reformatted problem
        if verbose:
            raw_output, full_prompt = llm_client.generate_semi_synthetic_problem(
                model=model,
                api_key=api_key,
                system_prompt=system_prompt,
                seed_problem=seed_problem,
                seed_solution=seed_solution,
                return_prompt=True,
            )
        else:
            raw_output = llm_client.generate_semi_synthetic_problem(
                model=model,
                api_key=api_key,
                system_prompt=system_prompt,
                seed_problem=seed_problem,
                seed_solution=seed_solution,
            )
            full_prompt = None
        
        # Parse LLM output
        parsed_sections = parser.parse_llm_output(raw_output, parse_code_subsections=parse_code_subsections)
        
        # Build problem data
        problem_id = f"p{problem_idx}"
        today = date.today().isoformat()
        
        # Compute and store pair hash for tracking
        pair_hash = compute_pair_hash(arxiv_id, seed_problem, seed_solution)
        
        # Build original seed structure
        original_seed = {
            "seed_type": seed_type,
            "arxiv_id": arxiv_id,
            "seed_data": {}
        }
        
        if seed_type == "matched":
            original_seed["seed_data"]["problem"] = seed_problem
            original_seed["seed_data"]["solution"] = seed_solution
        elif seed_type == "example":
            # For example_pairs, the 'problem' field contains the example
            original_seed["seed_data"]["example"] = seed_data_dict.get("example", seed_problem) if seed_data_dict else seed_problem
        
        problem_data = {
            "problem_id": problem_id,
            "problem_metadata": {
                "Public problem": "yes",
                "Auto-verifiable": "no",
                "Domain of theoretical physics": topic_name,
                "Difficulty level": "",  # Can be left empty or inferred
                "Topic Entry ID": "",  # Not applicable for semi-synthetic
                "Authors": "",
                "Reviewers": "",
                "Novelty": "",
                "Problem ID": f"Problem {problem_idx}",
                "Problem Version": "",
                "Variation of a different problem": "",
                "Problem origin": f"{model} (semi-synthetic)",
                "Date problem was added to the data set": today,
                "Author comments": f"Semi-synthetic problem {problem_idx} from {topic_name} dataset, source: {arxiv_id}",
                "Problem Description": parsed_sections.get("Problem Description", ""),
                "Source": f"{arxiv_id}",  # Store arxiv ID
                "Seed Pair Hash": pair_hash,  # Store hash for tracking processed pairs
            },
            "problem_details": {
                "Problem Statement": parsed_sections.get("Problem Statement", ""),
                "Solution": parsed_sections.get("Solution", ""),
                "Answer Requirements": _wrap_code_section(parsed_sections.get("Answer Requirements", "")),
                "Answer": parsed_sections.get("Answer", ""),
                "Code": _wrap_code_section(parsed_sections.get("Code", "")),
            },
            "original_seed": original_seed,  # NEW
        }
        
        # Save problem file
        filepath = out_dir / f"p{problem_idx}.json"
        io_utils.save_problem_json(problem_data, filepath)
        
        # Generate test cases for the problem
        try:
            generate_test_cases.generate_test_cases_for_problem(str(filepath))
        except Exception as e:
            # Log error but don't fail problem generation
            print(f"Warning: Failed to generate test cases for {filepath}: {e}")
        
        # Save prompt and raw output if verbose
        if verbose:
            if full_prompt:
                prompt_filepath = out_dir / f"p{problem_idx}_prompt.txt"
                with open(prompt_filepath, "w", encoding="utf-8") as f:
                    f.write(full_prompt)
            if raw_output:
                raw_output_filepath = out_dir / f"p{problem_idx}_raw_output.txt"
                with open(raw_output_filepath, "w", encoding="utf-8") as f:
                    f.write(raw_output)
        
        return problem_data
        
    except Exception as e:
        print(f"Error processing problem p{problem_idx} from {arxiv_id}: {e}")
        return None


def worker_process_seed_pairs(
    seed_pairs_with_indices: List[Tuple[str, str, str, int, str, Dict]],  # Updated: (arxiv_id, problem, solution, problem_idx, seed_type, seed_dict)
    system_prompt_template: str,
    out_dir_str: str,  # Path as string for multiprocessing compatibility
    api_key: str,
    model: str,
    topic_name: str,
    verbose: bool,
    parse_code_subsections: bool,
    progress_counter: Optional[Any] = None,
    progress_lock: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """
    Worker function that processes a batch of seed pairs.
    
    Args:
        seed_pairs_with_indices: List of (arxiv_id, problem, solution, problem_idx, seed_type, seed_dict) tuples
        system_prompt_template: System prompt template
        out_dir_str: Output directory as string (converted to Path inside)
        api_key: API key
        model: Model name
        topic_name: Topic name
        verbose: Whether to save prompts
        parse_code_subsections: Whether to parse code subsections
        progress_counter: Shared counter for progress tracking (optional)
        progress_lock: Lock for thread-safe counter updates (optional)
        
    Returns:
        List of generated problem data dictionaries
    """
    out_dir = Path(out_dir_str)  # Convert string back to Path
    generated_problems = []
    
    for arxiv_id, problem, solution, problem_idx, seed_type, seed_dict in seed_pairs_with_indices:
        result = process_seed_pair(
            seed_problem=problem,
            seed_solution=solution,
            arxiv_id=arxiv_id,
            problem_idx=problem_idx,
            system_prompt_template=system_prompt_template,
            out_dir=out_dir,
            api_key=api_key,
            model=model,
            topic_name=topic_name,
            verbose=verbose,
            parse_code_subsections=parse_code_subsections,
            seed_type=seed_type,  # NEW
            seed_data_dict=seed_dict,  # NEW
        )
        
        if result:
            generated_problems.append(result)
            # Update shared progress counter if provided
            if progress_counter is not None and progress_lock is not None:
                with progress_lock:
                    progress_counter.value += 1
    
    return generated_problems


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    cfg = load_config(config_path)
    
    dataset_name: str = cfg["dataset_name"]
    seed_data_path: str = cfg["seed_data_path"]
    model: str = cfg.get("model", "gemini-2.5-pro")
    verbose: bool = cfg.get("verbose", False)
    num_problems: Optional[int] = cfg.get("num_problems", None)  # None means process all
    parse_code_subsections: bool = cfg.get("parse_code_subsections", False)  # Default: False (single code block)
    num_workers: int = int(cfg.get("num_workers", 8))  # Number of parallel workers
    
    # Handle topic configuration: support both new path-based and old topic-based approach
    repo_root = Path(__file__).resolve().parents[1]
    
    if "topic_list_path" in cfg and "system_prompt_path" in cfg:
        # New path-based approach
        topic_list_path = Path(cfg["topic_list_path"])
        system_prompt_path = Path(cfg["system_prompt_path"])
        
        # Resolve relative paths to absolute (relative to repo root)
        if not topic_list_path.is_absolute():
            topic_list_path = (repo_root / topic_list_path).resolve()
        else:
            topic_list_path = topic_list_path.resolve()
        if not system_prompt_path.is_absolute():
            system_prompt_path = (repo_root / system_prompt_path).resolve()
        else:
            system_prompt_path = system_prompt_path.resolve()
        
        # Derive topic_name from path or use provided one
        topic_name = cfg.get("topic_name", None)
        if topic_name is None:
            # Try to derive from topic_list_path if it exists, otherwise from system_prompt_path
            if topic_list_path.exists() and topic_list_path.name == "topic_list.json":
                topic_name = topic_list_path.parent.name
            elif topic_list_path.exists():
                topic_name = topic_list_path.stem
            else:
                # Derive from system_prompt_path
                topic_name = system_prompt_path.parent.name
    elif "system_prompt_path" in cfg:
        # Semi-synthetic: only system_prompt_path required (topic_list_path optional)
        system_prompt_path = Path(cfg["system_prompt_path"])
        if not system_prompt_path.is_absolute():
            system_prompt_path = (repo_root / system_prompt_path).resolve()
        else:
            system_prompt_path = system_prompt_path.resolve()
        
        topic_name = cfg.get("topic_name", system_prompt_path.parent.name)
        
        # topic_list_path is optional for semi-synthetic
        if "topic_list_path" in cfg:
            topic_list_path = Path(cfg["topic_list_path"])
            if not topic_list_path.is_absolute():
                topic_list_path = (repo_root / topic_list_path).resolve()
            else:
                topic_list_path = topic_list_path.resolve()
        else:
            topic_list_path = None
    elif "topic" in cfg:
        # Backward compatibility: old topic-based approach
        topic_name = cfg["topic"]
        # topic_list_path is optional for semi-synthetic
        topic_list_path_candidate = (repo_root / "topics" / topic_name / "topic_list.json").resolve()
        if topic_list_path_candidate.exists():
            topic_list_path = topic_list_path_candidate
        else:
            topic_list_path = None
        # Check prompts directory first, then fall back to topics directory
        prompts_path = (repo_root / "prompts" / topic_name / "system_prompt.txt").resolve()
        topics_path = (repo_root / "topics" / topic_name / "system_prompt.txt").resolve()
        if prompts_path.exists():
            system_prompt_path = prompts_path
        elif topics_path.exists():
            system_prompt_path = topics_path
        else:
            raise FileNotFoundError(f"System prompt not found in prompts/{topic_name}/ or topics/{topic_name}/")
    else:
        raise ValueError("Config must specify 'system_prompt_path' (required), optionally 'topic_list_path', or 'topic' (deprecated)")
    
    seed = cfg.get("seed", None)
    rng = np.random.default_rng(seed)
    
    # Validate paths exist (topic_list_path is optional for semi-synthetic)
    if topic_list_path is not None and not topic_list_path.exists():
        raise FileNotFoundError(f"Topic list file not found: {topic_list_path}")
    if not system_prompt_path.exists():
        raise FileNotFoundError(f"System prompt file not found: {system_prompt_path}")
    
    with open(system_prompt_path, "r", encoding="utf-8") as f:
        system_prompt_template = f.read()
    
    # Resolve seed data path (can be relative to repo root or absolute)
    seed_base_path = Path(seed_data_path)
    if not seed_base_path.is_absolute():
        seed_base_path = repo_root / seed_data_path
    
    if not seed_base_path.exists():
        raise FileNotFoundError(f"Seed data path not found: {seed_base_path}")
    
    # Prepare output directory (config can override with data_type: adapted_data)
    data_type: str = cfg.get("data_type", "semi_synthetic_data")
    out_dir = io_utils.make_dataset_dir(topic_name, dataset_name, base=repo_root, data_type=data_type)
    
    if not args.dry_run:
        io_utils.save_config_copy_from_path(config_path, out_dir)

    api_key = get_api_key() if not args.dry_run else ""

    # Load existing problems to determine starting index
    existing_problems = io_utils.load_existing_problems(out_dir) if not args.dry_run else []
    max_existing_index = io_utils.get_max_problem_index(out_dir)
    start_index = max_existing_index + 1
    
    # Load hashes of already processed pairs
    processed_hashes = load_processed_pair_hashes(out_dir) if not args.dry_run else set()
    if processed_hashes:
        print(f"Found {len(processed_hashes)} already processed seed pairs")
    
    # Collect all seed pairs with their hashes
    all_seed_pairs: List[tuple[str, str, str, str, str, Dict]] = []  # Updated: (arxiv_id, problem, solution, pair_hash, seed_type, seed_dict)
    
    print(f"Scanning seed data directories in {seed_base_path}...")
    for seed_dir in seed_loader.find_seed_directories(seed_base_path):
        arxiv_id = seed_loader.get_arxiv_id_from_path(seed_dir)
        example_pairs, matched_pairs = seed_loader.load_seed_pairs(seed_dir)
        
        # Process example pairs
        for pair in example_pairs:
            problem = pair.get("problem", "")
            solution = pair.get("solution", "")
            # Ensure problem and solution are strings (handle cases where they might be dicts)
            if isinstance(problem, dict):
                problem = str(problem)
            elif not isinstance(problem, str):
                problem = ""
            if isinstance(solution, dict):
                solution = str(solution)
            elif not isinstance(solution, str):
                solution = ""
            if problem and solution:
                pair_hash = compute_pair_hash(arxiv_id, problem, solution)
                all_seed_pairs.append((arxiv_id, problem, solution, pair_hash, "example", pair))  # NEW: seed_type and full dict
        
        # Process matched pairs
        for pair in matched_pairs:
            problem = pair.get("problem", "")
            solution = pair.get("solution", "")
            # Ensure problem and solution are strings (handle cases where they might be dicts)
            if isinstance(problem, dict):
                problem = str(problem)
            elif not isinstance(problem, str):
                problem = ""
            if isinstance(solution, dict):
                solution = str(solution)
            elif not isinstance(solution, str):
                solution = ""
            if problem and solution:
                pair_hash = compute_pair_hash(arxiv_id, problem, solution)
                all_seed_pairs.append((arxiv_id, problem, solution, pair_hash, "matched", pair))  # NEW: seed_type and full dict
    
    total_pairs = len(all_seed_pairs)
    print(f"Found {total_pairs} seed problem-solution pairs")

    # Filter out already processed pairs using hash tracking
    already_processed_count = 0
    if processed_hashes:
        original_count = len(all_seed_pairs)
        all_seed_pairs = [(a, p, s, h, st, sd) for a, p, s, h, st, sd in all_seed_pairs if h not in processed_hashes]
        already_processed_count = original_count - len(all_seed_pairs)
        if already_processed_count > 0:
            print(f"Already processed: {already_processed_count} pairs (identified by hash)")

    # num_problems is the TOTAL target. Compute how many new ones to add.
    if num_problems is not None and num_problems > 0:
        new_to_add = max(0, num_problems - already_processed_count)
        if new_to_add == 0:
            print(f"Target {num_problems} already met ({already_processed_count} existing). Nothing to do.")
            if args.dry_run:
                print(f"[dry-run] Output directory: {out_dir}")
            return
        all_seed_pairs = all_seed_pairs[:new_to_add]
        print(f"Adding {new_to_add} new pairs to reach target of {num_problems}")
    
    pairs_to_process = len(all_seed_pairs)
    
    if pairs_to_process == 0:
        print("No seed pairs to process. Exiting.")
        if args.dry_run:
            print(f"[dry-run] Would process 0 pairs")
            print(f"[dry-run] Starting from problem index {start_index}")
            print(f"[dry-run] Output directory: {out_dir}")
        return
    
    # Validate num_workers
    if num_workers < 1:
        raise ValueError(f"num_workers must be >= 1, got {num_workers}")
    
    # Prepare seed pairs with assigned problem indices
    seed_pairs_with_indices: List[Tuple[str, str, str, int, str, Dict]] = []
    for idx, (arxiv_id, problem, solution, pair_hash, seed_type, seed_dict) in enumerate(all_seed_pairs):
        problem_idx = start_index + idx
        seed_pairs_with_indices.append((arxiv_id, problem, solution, problem_idx, seed_type, seed_dict))
    
    # Distribute work across workers (round-robin)
    worker_assignments: List[List[Tuple[str, str, str, int, str, Dict]]] = [[] for _ in range(num_workers)]
    for idx, pair_with_idx in enumerate(seed_pairs_with_indices):
        worker_idx = idx % num_workers
        worker_assignments[worker_idx].append(pair_with_idx)
    
    if args.dry_run:
        print(f"[dry-run] Would process {pairs_to_process} pairs")
        print(f"[dry-run] Starting from problem index {start_index}")
        print(f"[dry-run] Output directory: {out_dir}")
        print(f"[dry-run] Number of workers: {num_workers}")
        print(f"[dry-run] Work distribution:")
        for worker_idx, assignments in enumerate(worker_assignments, 1):
            print(f"[dry-run]   Worker {worker_idx}: {len(assignments)} pairs")
        return
    
    # Progress bar
    progress_bar = tqdm(
        total=pairs_to_process,
        desc="Processing seed pairs",
        unit="pair",
    )
    
    print(f"Processing {pairs_to_process} seed pairs using {num_workers} workers.")
    
    # Process with multiprocessing
    if num_workers == 1:
        # Single worker - no multiprocessing needed
        generated = worker_process_seed_pairs(
            seed_pairs_with_indices=worker_assignments[0],
            system_prompt_template=system_prompt_template,
            out_dir_str=str(out_dir),
            api_key=api_key,
            model=model,
            topic_name=topic_name,
            verbose=verbose,
            parse_code_subsections=parse_code_subsections,
        )
        generated_count = len(generated)
        progress_bar.update(generated_count)
    else:
        # Multiple workers - use shared counter for real-time progress tracking
        manager = Manager()
        progress_counter = manager.Value('i', 0)  # Shared integer counter
        progress_lock = manager.Lock()  # Lock for thread-safe increments
        last_progress = 0
        
        def update_progress():
            """Background thread to update progress bar from shared counter"""
            nonlocal last_progress
            while True:
                current = progress_counter.value
                if current > last_progress:
                    progress_bar.update(current - last_progress)
                    last_progress = current
                if current >= pairs_to_process:
                    break
                time.sleep(0.1)  # Update every 100ms
        
        # Start progress update thread
        progress_thread = threading.Thread(target=update_progress, daemon=True)
        progress_thread.start()
        
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Submit all worker tasks
            futures = []
            for worker_idx in range(num_workers):
                if worker_assignments[worker_idx]:  # Only submit if worker has assignments
                    future = executor.submit(
                        worker_process_seed_pairs,
                        seed_pairs_with_indices=worker_assignments[worker_idx],
                        system_prompt_template=system_prompt_template,
                        out_dir_str=str(out_dir),  # Convert Path to string for multiprocessing
                        api_key=api_key,
                        model=model,
                        topic_name=topic_name,
                        verbose=verbose,
                        parse_code_subsections=parse_code_subsections,
                        progress_counter=progress_counter,
                        progress_lock=progress_lock,
                    )
                    futures.append(future)
            
            # Wait for all workers to complete
            all_generated = []
            for future in as_completed(futures):
                try:
                    worker_results = future.result()
                    all_generated.extend(worker_results)
                except Exception as e:
                    print(f"Worker error: {e}")
            
            # Wait for progress thread to finish
            progress_thread.join(timeout=1.0)
        
        generated_count = len(all_generated)
    
    progress_bar.close()
    print(f"Generation complete! Generated {generated_count} problems from {pairs_to_process} seed pairs.")


if __name__ == "__main__":
    main()

