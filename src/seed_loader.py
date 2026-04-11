from __future__ import annotations

import yaml
from pathlib import Path
from typing import List, Dict, Any, Iterator, Tuple


def load_yaml_pairs(filepath: Path) -> List[Dict[str, Any]]:
    """
    Load YAML documents from a file (handles multi-document YAML files).
    
    Args:
        filepath: Path to the YAML file (e.g., example_pairs.yaml or matched_pairs.yaml)
        
    Returns:
        List of dictionaries, one per YAML document in the file
    """
    if not filepath.exists():
        return []
    
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            # Load all documents from multi-document YAML
            documents = list(yaml.safe_load_all(f))
            # Filter out None values (empty documents)
            return [doc for doc in documents if doc is not None]
    except (yaml.YAMLError, IOError) as e:
        # Return empty list on error
        return []


def get_arxiv_id_from_path(path: Path) -> str:
    """
    Extract arxiv ID from directory path.
    
    Args:
        path: Path to an arxiv ID directory (e.g., .../0704.1040v1/)
        
    Returns:
        Arxiv ID string (e.g., "0704.1040v1")
    """
    return path.name


def find_seed_directories(base_path: Path) -> Iterator[Path]:
    """
    Iterate over seed directories in the base path.
    
    A seed directory is any subdirectory that contains either:
    - matched_pairs.yaml, or
    - example_pairs.yaml, or
    - matches the arxiv ID pattern (contains digits and dots)
    
    Args:
        base_path: Base directory containing seed subdirectories
        
    Yields:
        Path objects for each seed directory
    """
    if not base_path.exists():
        return
    
    # Iterate over all subdirectories
    for item in base_path.iterdir():
        if item.is_dir():
            # Check if it contains seed data files OR looks like an arxiv ID
            has_matched_pairs = (item / "matched_pairs.yaml").exists()
            has_example_pairs = (item / "example_pairs.yaml").exists()
            looks_like_arxiv = any(c.isdigit() for c in item.name) and '.' in item.name
            
            if has_matched_pairs or has_example_pairs or looks_like_arxiv:
                yield item


def load_seed_pairs(seed_dir: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Load example_pairs.yaml and matched_pairs.yaml from a seed directory.
    
    Args:
        seed_dir: Path to an arxiv ID directory containing the YAML files
        
    Returns:
        Tuple of (example_pairs, matched_pairs) where each is a list of dictionaries
    """
    example_pairs_path = seed_dir / "example_pairs.yaml"
    matched_pairs_path = seed_dir / "matched_pairs.yaml"
    
    example_pairs = load_yaml_pairs(example_pairs_path)
    matched_pairs = load_yaml_pairs(matched_pairs_path)
    
    return example_pairs, matched_pairs

