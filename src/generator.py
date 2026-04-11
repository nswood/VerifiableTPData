from __future__ import annotations

import argparse
import re
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date
from multiprocessing import Manager, Value
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml
from tqdm import tqdm

from . import topics as topics_mod
from . import io_utils
from . import llm_client
from . import parser
from . import generate_test_cases

try:
    # genai.py contains all Gemini/GenAI configuration and functions
    from genai import get_api_key  # type: ignore
except Exception:  # pragma: no cover - robust import fallback
    def get_api_key() -> str:  # type: ignore
        return ""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate synthetic problems from LLMs")
    p.add_argument("--config", type=str, required=True, help="Path to YAML config")
    p.add_argument("--dry-run", action="store_true", help="Sample topics only; no writes")
    return p.parse_args()


def load_config(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


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


def _build_standard_sections() -> str:
    """
    Build standard section instructions for the system prompt.
    
    Returns:
        String containing standard section format
    """
    return """
\\section{Answer Requirements}

- Provide a separate Python function skeleton for each task that students will implement.
- Each task must have its own distinct function with a unique name.
- Include likely imports (math, cmath, numpy, itertools, etc.) even if not all are used.
- Each function's parameters must match the problem inputs for that task.
- Docstring must describe:
  - expected return type (float, complex, bool, str, or tuple),
  - allowed categorical options if applicable.
- Functions for later tasks may call functions from earlier tasks if needed for the implementation.

\\section{Solution}

- Give the full derivation or conceptual reasoning leading to the result(s).
- Include any algebraic steps, integrals, or group-theory logic as needed.
- May include equations in LaTeX; clarity is more important than brevity.

\\section{Answer}

- List each task's final result concisely.
- Numeric answers should be explicit constants or evaluable expressions.
- Categorical answers should be quoted strings or items from a small known set.

\\section{Code}

- Provide a separate subsection for each task (e.g., \\subsection{Task 1}, \\subsection{Task 2}, etc.).
- Each subsection must contain a working Python implementation of the corresponding function from the Answer Requirements section.
- Each implementation must return the correct numeric or categorical answers for that task.
- You may let Python perform the numeric evaluation (no manual arithmetic required).
- Functions may call previous task functions if needed (e.g., Task 2's function may call Task 1's function).
- Only return the updated python function(s) with no additional commentary.
""".strip()


def _build_prompt_for_problem(
    system_prompt: str,
    topic_entry: Dict[str, Any],
    previous_descriptions: Optional[List[str]] = None,
) -> str:
    """
    Build a prompt for a single problem (extracted from llm_client logic).
    
    Args:
        system_prompt: System prompt template
        topic_entry: Dictionary containing topic information
        previous_descriptions: Optional list of previous problem descriptions from the same topic
        
    Returns:
        Full prompt string combining system and user prompts
    """
    # Build the user prompt with topic information
    topic_title = topic_entry.get("topic", "")
    topic_description = topic_entry.get("description", "")
    
    user_prompt = f"Generate a new QFT exercise in this format for the topic:\n\"{topic_title}\""
    if topic_description:
        user_prompt += f"\n\nTopic description: {topic_description}"
    
    # Add previous problem descriptions if provided
    if previous_descriptions:
        user_prompt += "\n\nPreviously generated problems in this topic:"
        for i, desc in enumerate(previous_descriptions, 1):
            user_prompt += f"\n{i}. {desc}"
        user_prompt += "\n\nIMPORTANT: Generate a problem with tasks that are different from those listed above. Some similarity is acceptable, but the problem should not be an exact copy or nearly identical to any of the previous problems."
    
    # Combine system prompt and user prompt
    full_prompt = f"{system_prompt}\n\n{user_prompt}"
    return full_prompt


def assign_problems_to_topics(
    topic_ids: List[str],
    level_topics: Dict[str, List[Dict[str, Any]]],
    existing_assignment: Optional[Dict[str, List[int]]] = None,
    start_index: int = 1,
) -> Dict[str, List[int]]:
    """
    Assign exactly one problem to each specified topic ID.
    
    Args:
        topic_ids: List of topic IDs to generate problems for (one problem per topic)
        level_topics: Dictionary mapping level names to lists of topic entries
        existing_assignment: Optional existing assignment dictionary when resuming
        start_index: Starting problem index (for resuming)
        
    Returns:
        Dictionary mapping topic_entry_id to list of problem IDs (one ID per topic)
    """
    # Validate that all topic IDs exist
    all_topic_ids = set()
    for topics_list in level_topics.values():
        for topic_entry in topics_list:
            topic_id = topic_entry.get("id")
            if topic_id:
                all_topic_ids.add(topic_id)
    
    invalid_ids = [tid for tid in topic_ids if tid not in all_topic_ids]
    if invalid_ids:
        raise ValueError(f"Invalid topic IDs: {invalid_ids}. Available topic IDs: {sorted(all_topic_ids)}")
    
    # Initialize assignment
    assignment: Dict[str, List[int]] = {}
    if existing_assignment:
        assignment = {k: v.copy() for k, v in existing_assignment.items()}
    
    # Determine starting problem ID
    if existing_assignment:
        existing_count = sum(len(problem_ids) for problem_ids in existing_assignment.values())
        next_problem_id = start_index
    else:
        next_problem_id = start_index
    
    # Assign one problem to each topic ID
    for topic_id in topic_ids:
        # Skip if already assigned (when resuming)
        if topic_id in assignment and assignment[topic_id]:
            continue
        
        if topic_id not in assignment:
            assignment[topic_id] = []
        assignment[topic_id].append(next_problem_id)
        next_problem_id += 1
    
    return assignment


def distribute_problems(
    num_problems: int,
    level_split: Dict[str, float],
    level_topics: Dict[str, List[Dict[str, Any]]],
    existing_assignment: Optional[Dict[str, List[int]]] = None,
    start_index: int = 1,
) -> Dict[str, List[int]]:
    """
    Deterministically distribute N problems across levels and topics based on split ratios.

    Problems are allocated to levels proportionally, then round-robined across
    topics within each level for even coverage.

    Args:
        num_problems: Total number of problems to distribute
        level_split: Dictionary mapping level names to split ratios (e.g., 0.25 each)
        level_topics: Dictionary mapping level names to lists of topic entries
        existing_assignment: Optional existing assignment dictionary when resuming
        start_index: Starting problem index (for resuming)

    Returns:
        Dictionary mapping topic_entry_id (e.g., "PG-21") to list of problem IDs
    """
    # Collect levels that exist in both split config and topic list
    level_keys: List[str] = []
    level_ratios: List[float] = []
    for level in level_topics:
        if level in level_split:
            level_keys.append(level)
            level_ratios.append(level_split[level])

    if not level_keys:
        return {}

    # Normalize ratios to sum to 1
    total = sum(level_ratios)
    if total > 0:
        level_ratios = [r / total for r in level_ratios]
    else:
        level_ratios = [1.0 / len(level_keys)] * len(level_keys)

    # Initialize assignment
    assignment: Dict[str, List[int]] = {}
    if existing_assignment:
        assignment = {k: v.copy() for k, v in existing_assignment.items()}

    # Determine remaining problems
    if existing_assignment:
        existing_count = sum(len(pids) for pids in existing_assignment.values())
        remaining = num_problems - existing_count
    else:
        remaining = num_problems

    if remaining <= 0:
        return assignment

    next_problem_id = start_index

    # Compute exact counts per level using largest-remainder method
    exact_counts = [remaining * r for r in level_ratios]
    floor_counts = [int(c) for c in exact_counts]
    remainders = [exact_counts[i] - floor_counts[i] for i in range(len(level_keys))]

    # Distribute leftover problems to levels with largest fractional remainders
    leftover = remaining - sum(floor_counts)
    indices_by_remainder = sorted(range(len(level_keys)), key=lambda i: remainders[i], reverse=True)
    for i in range(leftover):
        floor_counts[indices_by_remainder[i]] += 1

    level_counts = {level_keys[i]: floor_counts[i] for i in range(len(level_keys))}

    # Assign problems within each level: round-robin across topics
    for level in level_keys:
        count = level_counts[level]
        if count == 0:
            continue

        topics_at_level = level_topics.get(level, [])
        topic_ids_at_level = [t.get("id") for t in topics_at_level if t.get("id")]
        if not topic_ids_at_level:
            continue

        for i in range(count):
            topic_id = topic_ids_at_level[i % len(topic_ids_at_level)]
            if topic_id not in assignment:
                assignment[topic_id] = []
            assignment[topic_id].append(next_problem_id)
            next_problem_id += 1

    return assignment


def get_topic_entry_by_id(
    topic_id: str,
    level_topics: Dict[str, List[Dict[str, Any]]],
) -> Optional[Dict[str, Any]]:
    """Find a topic entry by its ID."""
    for topics_list in level_topics.values():
        for topic_entry in topics_list:
            if topic_entry.get("id") == topic_id:
                return topic_entry
    return None


def get_previous_descriptions_for_topic(
    topic_id: str,
    existing_problems: List[Dict[str, Any]],
    topic_name: str,
) -> List[str]:
    """Get previous problem descriptions for a given topic entry ID."""
    descriptions = []
    for problem in existing_problems:
        # Check if topic_entry_id matches
        problem_topic_id = problem.get("problem_metadata", {}).get("Topic Entry ID", "")
        if problem_topic_id == topic_id:
            desc = problem.get("problem_metadata", {}).get("Problem Description", "")
            if desc:
                descriptions.append(desc)
    return descriptions


def worker_process_problems(
    worker_assignments: List[Tuple[str, List[int]]],
    topic_list_path: Path,
    system_prompt_template: str,
    topic_name: str,
    out_dir: Path,
    api_key: str,
    model: str,
    task_min: int,
    task_max: int,
    tasks_dependent: bool,
    existing_problems_file: Optional[Path],
    verbose: bool,
    rng_seed: Optional[int],
    progress_counter: Optional[Any] = None,
    progress_lock: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """
    Worker function that processes assigned topics and generates problems.
    
    Args:
        worker_assignments: List of (topic_id, problem_ids) tuples for this worker
        topic_list_path: Path to topic_list.json file
        system_prompt_template: System prompt template
        topic_name: Name of the topic (e.g., "QFT") - used for output directory naming
        out_dir: Output directory
        api_key: API key
        model: Model name
        task_min: Minimum number of tasks
        task_max: Maximum number of tasks
        tasks_dependent: Whether tasks are dependent
        existing_problems_file: Path to existing problems directory (for loading context)
        verbose: Whether to save prompts
        rng_seed: Random seed for this worker
        
    Returns:
        List of generated problem data dictionaries
    """
    # Reload level_topics in worker (for multiprocessing compatibility)
    level_topics = topics_mod.load_topics(topic_list_path)
    
    # Load existing problems for context
    if existing_problems_file:
        worker_existing = io_utils.load_existing_problems(existing_problems_file)
    else:
        worker_existing = []

    generated_problems = []
    
    # Process each assigned topic sequentially
    for topic_id, problem_ids in worker_assignments:
        # Get topic entry
        topic_entry = get_topic_entry_by_id(topic_id, level_topics)
        if not topic_entry:
            continue
        
        # Get level for this topic entry
        level = None
        for lev, topics_list in level_topics.items():
            if any(t.get("id") == topic_id for t in topics_list):
                level = lev
                break
        
        if not level:
            continue
        
        topic_title = topic_entry.get("topic", "")
        domain = f"{topic_name}: {topic_title}" if topic_title else topic_name
        
        # Get previous descriptions for this topic
        previous_descriptions = get_previous_descriptions_for_topic(
            topic_id, worker_existing, topic_name
        )
        
        # Generate all problems for this topic
        for problem_idx in problem_ids:
            # Build task instruction text with range
            if task_min == task_max:
                task_count = task_min
                task_word = "task" if task_count == 1 else "tasks"
                if task_count == 1:
                    task_instruction = "List exactly one task; it should ask for a numeric or categorical result. The task may return multiple outputs, but you MUST only ask for one task."
                elif tasks_dependent:
                    task_instruction = f"List exactly {task_count} sequential {task_word}, where task n requires solving tasks 1 through n-1; each asks for a numeric or categorical result."
                else:
                    task_instruction = f"List exactly {task_count} {task_word}; each asks for a numeric or categorical result."
            else:
                if tasks_dependent:
                    task_instruction = f"List between {task_min} and {task_max} sequential tasks (choose the number that best fits the problem complexity), where task n requires solving tasks 1 through n-1; each asks for a numeric or categorical result."
                else:
                    task_instruction = f"List between {task_min} and {task_max} tasks (choose the number that best fits the problem complexity); each asks for a numeric or categorical result."

            system_prompt = system_prompt_template.replace("{{TASK_INSTRUCTIONS}}", task_instruction)
            system_prompt = system_prompt.replace("{{GRAPH_INSTRUCTIONS}}", "")
            standard_sections = _build_standard_sections()
            system_prompt = system_prompt.replace("{{STANDARD_SECTIONS}}", standard_sections)
            
            # Replace level placeholder
            system_prompt = system_prompt.replace("{{INSERT LEVEL}}", level)
            
            # Generate problem
            try:
                if verbose:
                    raw_output, full_prompt = llm_client.generate_problem(
                        model=model,
                        api_key=api_key,
                        system_prompt=system_prompt,
                        topic_entry=topic_entry,
                        seed=rng_seed,
                        previous_descriptions=previous_descriptions if previous_descriptions else None,
                        return_prompt=True,
                    )
                else:
                    raw_output = llm_client.generate_problem(
                        model=model,
                        api_key=api_key,
                        system_prompt=system_prompt,
                        topic_entry=topic_entry,
                        seed=rng_seed,
                        previous_descriptions=previous_descriptions if previous_descriptions else None,
                    )
                    full_prompt = None
            except Exception as e:
                print(f"Error generating problem p{problem_idx} for topic {topic_id}: {e}")
                continue
            
            parsed_sections = parser.parse_llm_output(raw_output)

            problem_id = f"p{problem_idx}"
            today = date.today().isoformat()

            problem_data = {
                "problem_id": problem_id,
                "problem_metadata": {
                    "Public problem": "yes",
                    "Auto-verifiable": "no",
                    "Domain of theoretical physics": domain,
                    "Difficulty level": level,
                    "Topic Entry ID": topic_id,
                    "Authors": "",
                    "Reviewers": "",
                    "Novelty": "",
                    "Problem ID": f"Problem {problem_idx}",
                    "Problem Version": "",
                    "Variation of a different problem": "",
                    "Problem origin": model,
                    "Date problem was added to the data set": today,
                    "Author comments": f"Problem {problem_idx} from {topic_name} dataset",
                    "Problem Description": parsed_sections.get("Problem Description", ""),
                },
                "problem_details": {
                    "Problem Statement": parsed_sections.get("Problem Statement", ""),
                    "Solution": parsed_sections.get("Solution", ""),
                    "Answer Requirements": _wrap_code_section(parsed_sections.get("Answer Requirements", "")),
                    "Answer": parsed_sections.get("Answer", ""),
                    "Code": _wrap_code_section(parsed_sections.get("Code", "")),
                },
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
            
            # Add to worker's existing problems for next iteration
            worker_existing.append(problem_data)
            generated_problems.append(problem_data)
            
            # Update shared progress counter if provided
            if progress_counter is not None and progress_lock is not None:
                # Use lock for thread-safe increment (Manager.Value needs explicit locking)
                with progress_lock:
                    progress_counter.value += 1
            
            # Update previous descriptions for next problem in same topic
            desc = problem_data.get("problem_metadata", {}).get("Problem Description", "")
            if desc:
                previous_descriptions.append(desc)
    
    return generated_problems


def run_topic_generation(args: argparse.Namespace) -> None:
    config_path = Path(args.config).resolve()
    cfg = load_config(config_path)

    dataset_name: str = cfg["dataset_name"]
    model: str = cfg.get("model", "gemini-2.5-pro")
    seed: Any = cfg.get("seed", None)
    verbose: bool = cfg.get("verbose", False)  # Enable verbose mode to log prompts
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
            # Try to derive from topic_list_path
            if topic_list_path.name == "topic_list.json":
                topic_name = topic_list_path.parent.name
            else:
                topic_name = topic_list_path.stem
    elif "topic" in cfg:
        # Backward compatibility: old topic-based approach
        topic_name = cfg["topic"]
        topic_list_path = (repo_root / "topics" / topic_name / "topic_list.json").resolve()
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
        raise ValueError("Config must specify either 'topic_list_path' and 'system_prompt_path', or 'topic' (deprecated)")
    
    # Check if topics are specified directly (direct assignment mode)
    topic_ids: Optional[List[str]] = cfg.get("topics", None)
    if topic_ids is not None:
        # Direct assignment mode: generate one problem per specified topic
        num_problems = len(topic_ids)
        level_split = {}  # Not used in direct assignment mode
    else:
        num_problems = int(cfg["num_problems"])
        # Support both "level_split" (new) and "level_probs" (backwards compat)
        level_split = dict(cfg.get("level_split", cfg.get("level_probs", {})))
    
    # Validate num_workers
    if num_workers < 1:
        raise ValueError(f"num_workers must be >= 1, got {num_workers}")

    # Task configuration parameters
    task_config = cfg.get("task_config", {})
    task_min = int(task_config.get("tasks_min", cfg.get("tasks_min", 1)))
    task_max = int(task_config.get("tasks_max", cfg.get("tasks_max", 3)))
    tasks_dependent = bool(task_config.get("tasks_dependent", cfg.get("tasks_dependent", False)))
    
    # Validate task configuration
    if task_min < 1:
        raise ValueError(f"tasks_min must be >= 1, got {task_min}")
    if task_max < task_min:
        raise ValueError(f"tasks_max ({task_max}) must be >= tasks_min ({task_min})")

    # Validate paths exist
    if not topic_list_path.exists():
        raise FileNotFoundError(f"Topic list file not found: {topic_list_path}")
    if not system_prompt_path.exists():
        raise FileNotFoundError(f"System prompt file not found: {system_prompt_path}")

    with open(system_prompt_path, "r", encoding="utf-8") as f:
        system_prompt_template = f.read()

    # Load and flatten topics, derive available levels dynamically
    level_topics = topics_mod.load_topics(topic_list_path)
    available_levels = list(level_topics.keys())
    
    # Validate level_split keys against available levels
    if topic_ids is None and level_split:
        invalid_levels = [k for k in level_split if k not in available_levels]
        if invalid_levels:
            raise ValueError(f"Unknown levels in level_split: {invalid_levels}. Available: {available_levels}")

    out_dir = io_utils.make_dataset_dir(topic_name, dataset_name, base=repo_root, data_type="synthetic_data")

    if not args.dry_run:
        io_utils.save_config_copy_from_path(config_path, out_dir)

    api_key = get_api_key() if not args.dry_run else ""

    # Load existing problems and assignment
    existing_problems = io_utils.load_existing_problems(out_dir) if not args.dry_run else []
    original_assignment = io_utils.load_assignment(out_dir)

    # assignment.json tracks which IDs have been reserved
    all_assigned_ids = set()
    if original_assignment:
        for problem_ids in original_assignment.values():
            all_assigned_ids.update(problem_ids)

    # Find which assigned IDs actually have files (including subdirs like qc_failed/)
    generated_ids = set()
    for json_file in out_dir.rglob("p*.json"):
        match = re.search(r'p(\d+)\.json$', json_file.name)
        if match:
            try:
                generated_ids.add(int(match.group(1)))
            except ValueError:
                continue

    # Count top-level JSON files for progress tracking
    initial_json_count = len(list(out_dir.glob("p*.json")))

    # Next ID starts after the highest assigned ID
    start_index = max(all_assigned_ids) + 1 if all_assigned_ids else 1

    # num_problems is the TOTAL target. Determine how many new ones to distribute.
    total_assigned = len(all_assigned_ids)
    problems_to_add = max(0, num_problems - total_assigned)

    if problems_to_add > 0:
        # Distribute new problems across topics
        new_assignment = distribute_problems(
            num_problems=problems_to_add,
            level_split=level_split,
            level_topics=level_topics,
            start_index=start_index,
        )

        # Merge with original assignment
        full_assignment = {k: v.copy() for k, v in original_assignment.items()} if original_assignment else {}
        for topic_id, problem_ids in new_assignment.items():
            if topic_id not in full_assignment:
                full_assignment[topic_id] = []
            full_assignment[topic_id].extend(problem_ids)
    else:
        full_assignment = original_assignment if original_assignment else {}

    # Filter to only problems that don't have files yet (checking subdirs too)
    assignment = {}
    for topic_id, problem_ids in full_assignment.items():
        remaining = [pid for pid in problem_ids if pid not in generated_ids]
        if remaining:
            assignment[topic_id] = remaining
    remaining_problems = sum(len(pids) for pids in assignment.values())

    if remaining_problems <= 0:
        if args.dry_run:
            print(f"[dry-run] No problems to generate (remaining: {remaining_problems}).")
        else:
            print(f"No problems to generate (remaining: {remaining_problems}).")
        return
    
    if args.dry_run:
        if topic_ids is not None:
            print(f"[dry-run] Direct topic assignment mode:")
            print(f"[dry-run] Topics specified: {len(topic_ids)}")
        else:
            print(f"[dry-run] Problem distribution sampled:")
        print(f"[dry-run] Total topics with assignments: {len(assignment)}")
        total_assigned = sum(len(problem_ids) for problem_ids in assignment.values())
        print(f"[dry-run] Total problems assigned: {total_assigned}")
        print(f"[dry-run] Problems to generate: {remaining_problems}")
        print(f"[dry-run] Number of workers: {num_workers}")
        print()
        
        # Show overall distribution
        print(f"[dry-run] Overall distribution by topic:")
        for topic_id, problem_ids in sorted(assignment.items()):
            if problem_ids:
                topic_entry = get_topic_entry_by_id(topic_id, level_topics)
                topic_title = topic_entry.get("topic", "Unknown") if topic_entry else "Unknown"
                print(f"[dry-run]   {topic_id:8s}: {len(problem_ids):3d} problems -> {problem_ids} ({topic_title[:50]})")
        print()
        
        # Show worker assignments
        work_items = [(topic_id, problem_ids) for topic_id, problem_ids in assignment.items() if problem_ids]
        worker_assignments: List[List[Tuple[str, List[int]]]] = [[] for _ in range(num_workers)]
        for idx, work_item in enumerate(work_items):
            worker_idx = idx % num_workers
            worker_assignments[worker_idx].append(work_item)
        
        print(f"[dry-run] Worker assignments:")
        print("=" * 80)
        for worker_idx, assignments in enumerate(worker_assignments, 1):
            if not assignments:
                print(f"[dry-run] Worker {worker_idx}: (no assignments)")
                continue
            
            total_problems = sum(len(problem_ids) for _, problem_ids in assignments)
            print(f"[dry-run] Worker {worker_idx}: {total_problems} problems across {len(assignments)} topics")
            
            for topic_id, problem_ids in assignments:
                topic_entry = get_topic_entry_by_id(topic_id, level_topics)
                topic_title = topic_entry.get("topic", "Unknown") if topic_entry else "Unknown"
                level = None
                for lev, topics_list in level_topics.items():
                    if any(t.get("id") == topic_id for t in topics_list):
                        level = lev
                        break
                
                print(f"[dry-run]   ├─ {topic_id:8s} ({level or 'Unknown':25s}): {len(problem_ids):3d} problems")
                print(f"[dry-run]   │  Topic: {topic_title[:60]}")
                print(f"[dry-run]   │  Problem IDs: {problem_ids}")
            
            if worker_idx < num_workers:
                print()
        print("=" * 80)
        return
    
    # Save assignment - always merge with existing, never overwrite
    if full_assignment is not None:
        # We have a full assignment (either new or merged), save it
        io_utils.save_assignment(out_dir, full_assignment)
    elif original_assignment:
        # We're using existing assignment without modification, preserve it
        # (assignment.json already exists and hasn't been changed)
        pass
    
    # Distribute work to workers
    # Convert assignment to list of (topic_id, problem_ids) tuples
    work_items = [(topic_id, problem_ids) for topic_id, problem_ids in assignment.items() if problem_ids]
    
    # Distribute work items across workers (round-robin)
    worker_assignments: List[List[Tuple[str, List[int]]]] = [[] for _ in range(num_workers)]
    for idx, work_item in enumerate(work_items):
        worker_idx = idx % num_workers
        worker_assignments[worker_idx].append(work_item)
    
    # Progress bar for problem generation
    progress_desc = "Resuming generation" if all_assigned_ids else "Generating problems"
    progress_bar = tqdm(
        total=remaining_problems,
        desc=progress_desc,
        unit="problem",
        initial=0,
    )
    
    if all_assigned_ids:
        print(f"Found {len(all_assigned_ids)} existing problems. Generating {remaining_problems} more (p{start_index}+).")
    else:
        print(f"Generating {num_problems} problems using {num_workers} workers.")
    
    # Process with multiprocessing
    if num_workers == 1:
        # Single worker - no multiprocessing needed
        generated = worker_process_problems(
            worker_assignments=worker_assignments[0],
            topic_list_path=topic_list_path,
            system_prompt_template=system_prompt_template,
            topic_name=topic_name,
            out_dir=out_dir,
            api_key=api_key,
            model=model,
            task_min=task_min,
            task_max=task_max,
            tasks_dependent=tasks_dependent,
            existing_problems_file=out_dir,
            verbose=verbose,
            rng_seed=seed,
            progress_counter=None,  # No shared counter needed for single worker
        )
        # Count actual JSON files written
        final_json_count = len(list(out_dir.glob("p*.json")))
        files_written = final_json_count - initial_json_count
        progress_bar.update(files_written)
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
                if current >= remaining_problems:
                    break
                time.sleep(0.1)  # Update every 100ms
        
        # Start progress update thread
        progress_thread = threading.Thread(target=update_progress, daemon=True)
        progress_thread.start()
        
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Submit all worker tasks
            futures = []
            for worker_idx in range(num_workers):
                worker_seed = seed + worker_idx if seed is not None else None
                future = executor.submit(
                    worker_process_problems,
                    worker_assignments=worker_assignments[worker_idx],
                    topic_list_path=topic_list_path,
                    system_prompt_template=system_prompt_template,
                    topic_name=topic_name,
                    out_dir=out_dir,
                    api_key=api_key,
                    model=model,
                    task_min=task_min,
                    task_max=task_max,
                    tasks_dependent=tasks_dependent,
                    existing_problems_file=out_dir,
                    verbose=verbose,
                    rng_seed=worker_seed,
                    progress_counter=progress_counter,
                    progress_lock=progress_lock,
                )
                futures.append(future)
            
            # Collect results as they complete (for error handling)
            for future in as_completed(futures):
                try:
                    result = future.result()
                    # Progress is already tracked via shared counter
                except Exception as e:
                    print(f"Worker error: {e}")
        
        # Wait for progress thread to finish
        progress_thread.join(timeout=1.0)
        # Final update to ensure progress bar is at 100%
        final_progress = progress_counter.value
        if final_progress > last_progress:
            progress_bar.update(final_progress - last_progress)
    
    progress_bar.close()
    # Count actual JSON files written
    final_json_count = len(list(out_dir.glob("p*.json")))
    files_written = final_json_count - initial_json_count
    print(f"Generation complete! Generated {files_written} problems (from {initial_json_count} to {final_json_count} total).")


def main() -> None:
    run_topic_generation(parse_args())


if __name__ == "__main__":
    main()


