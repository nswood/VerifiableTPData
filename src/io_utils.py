from __future__ import annotations

import json
import random
import re
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple


def make_dataset_dir(topic: str, dataset_name: str, base: Path, data_type: str = "data") -> Path:
    """
    Create output directory for a dataset.
    
    Args:
        topic: Topic name (e.g., "QFT")
        dataset_name: Dataset name
        base: Base repository root path
        data_type: Type of data directory - "synthetic_data", "adapted_data", or legacy "semi_synthetic_data" (default: "data" for backward compatibility)
        
    Returns:
        Path to the created directory
    """
    out_dir = base / data_type / topic / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def save_config_copy_from_path(config_path: Path, dest_dir: Path) -> Path:
    dest = dest_dir / "config.yaml"
    shutil.copyfile(config_path, dest)
    return dest


def save_problem_json(problem_data: Dict[str, Any], filepath: Path) -> None:
    """Save problem data as a JSON file matching the specified format."""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(problem_data, f, ensure_ascii=False, indent=2)


def load_existing_problems(data_dir: Path) -> List[Dict[str, Any]]:
    """
    Load all existing problem JSON files from a dataset directory.
    
    Args:
        data_dir: Path to the dataset directory containing p*.json files
        
    Returns:
        List of problem dictionaries loaded from JSON files
    """
    problems = []
    if not data_dir.exists():
        return problems
    
    # Find all p*.json files in the directory
    for json_file in sorted(data_dir.glob("p*.json")):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                problem_data = json.load(f)
                problems.append(problem_data)
        except (json.JSONDecodeError, IOError) as e:
            # Skip files that can't be parsed
            continue
    
    return problems


def get_max_problem_index(data_dir: Path) -> int:
    """
    Extract the maximum problem index from existing p*.json files.
    
    Args:
        data_dir: Path to the dataset directory containing p*.json files
        
    Returns:
        Maximum problem index found, or 0 if no problems exist
    """
    if not data_dir.exists():
        return 0
    
    max_index = 0
    # Find all p*.json files (recursive - covers files moved to qc_failed/, etc.)
    for json_file in data_dir.rglob("p*.json"):
        # Extract number from filename like "p1.json", "p42.json", etc.
        match = re.search(r'p(\d+)\.json$', json_file.name)
        if match:
            try:
                index = int(match.group(1))
                max_index = max(max_index, index)
            except ValueError:
                # Skip if conversion fails
                continue
    
    return max_index


def save_assignment(data_dir: Path, assignment: Dict[str, List[int]]) -> None:
    """
    Save the problem assignment dictionary to assignment.json.
    
    Args:
        data_dir: Path to the dataset directory
        assignment: Dictionary mapping topic_entry_id to list of problem IDs
    """
    assignment_path = data_dir / "assignment.json"
    with open(assignment_path, "w", encoding="utf-8") as f:
        json.dump(assignment, f, ensure_ascii=False, indent=2)


def load_assignment(data_dir: Path) -> Optional[Dict[str, List[int]]]:
    """
    Load the problem assignment dictionary from assignment.json.
    
    Args:
        data_dir: Path to the dataset directory
        
    Returns:
        Assignment dictionary if it exists, None otherwise
    """
    assignment_path = data_dir / "assignment.json"
    if not assignment_path.exists():
        return None
    
    try:
        with open(assignment_path, "r", encoding="utf-8") as f:
            assignment = json.load(f)
            # Convert string keys to int lists if needed (for JSON compatibility)
            return {k: [int(x) for x in v] if isinstance(v, list) else v 
                    for k, v in assignment.items()}
    except (json.JSONDecodeError, IOError, ValueError):
        return None


def get_completed_problems_by_topic(data_dir: Path) -> Dict[str, List[int]]:
    """
    Extract which problems have been generated for each topic entry ID.
    
    Args:
        data_dir: Path to the dataset directory
        
    Returns:
        Dictionary mapping topic_entry_id to list of completed problem IDs
    """
    completed = {}
    if not data_dir.exists():
        return completed
    
    # Find all p*.json files and extract topic entry IDs
    for json_file in sorted(data_dir.glob("p*.json")):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                problem_data = json.load(f)
            
            # Extract problem ID from filename
            match = re.search(r'p(\d+)\.json$', json_file.name)
            if not match:
                continue
            problem_id = int(match.group(1))
            
            # Extract topic entry ID from domain metadata
            # Format: "Topic: subtopic" - we need to find the topic entry ID
            # We'll need to match this back to the topic entry ID through the problem metadata
            # For now, we'll store a mapping that can be resolved later
            domain = problem_data.get("problem_metadata", {}).get("Domain of theoretical physics", "")
            # The domain format is "Topic: subtopic", but we need the topic_entry ID
            # We'll need to match this in the generator when loading
            
        except (json.JSONDecodeError, IOError, ValueError):
            continue

    return completed


def split_train_val(
    source_dir: Path,
    train_ratio: float = 0.8,
    val_size: Optional[int] = None,
    seed: int = 42,
) -> Tuple[List[Path], List[Path]]:
    """
    Split problem JSON files into train/val subdirectories.

    Args:
        source_dir: Directory containing problem JSON files
        train_ratio: Fraction for train split (default 0.8, ignored if val_size set)
        val_size: Fixed number of val files (overrides train_ratio)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_files, val_files) paths in the new directories
    """
    all_files = sorted(source_dir.glob("*.json"))
    problem_files = [f for f in all_files if f.name not in ("assignment.json", "config.json")]

    if not problem_files:
        print(f"No problem files found in {source_dir}")
        return [], []

    random.seed(seed)
    shuffled = problem_files.copy()
    random.shuffle(shuffled)

    if val_size is not None:
        val_size = min(val_size, len(shuffled))
        val_files = shuffled[:val_size]
        train_files = shuffled[val_size:]
    else:
        split_idx = int(len(shuffled) * train_ratio)
        train_files = shuffled[:split_idx]
        val_files = shuffled[split_idx:]

    train_dir = source_dir / "train"
    val_dir = source_dir / "val"
    train_dir.mkdir(exist_ok=True)
    val_dir.mkdir(exist_ok=True)

    for f in train_files:
        shutil.copy2(f, train_dir / f.name)
    for f in val_files:
        shutil.copy2(f, val_dir / f.name)

    print(f"Split {len(problem_files)} files: train={len(train_files)}, val={len(val_files)}")
    print(f"  Train: {train_dir}")
    print(f"  Val: {val_dir}")

    return train_files, val_files

