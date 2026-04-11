import json
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np


def load_topics(topic_list_path: Path) -> Dict[str, List[dict]]:
    """
    Load topic_list.json from the given path and flatten subcategories
    into a dict mapping level -> list of topic entries (uniform within level).
    
    Args:
        topic_list_path: Path to topic_list.json file (can be a file path or directory path)
    """
    # Handle both file path and directory path for backward compatibility
    if topic_list_path.is_dir():
        json_path = topic_list_path / "topic_list.json"
    else:
        json_path = topic_list_path
    
    with open(json_path, "r", encoding="utf-8") as f:
        spec = json.load(f)

    flat: Dict[str, List[dict]] = {}
    for level, subdict in spec.items():
        bucket: List[dict] = []
        if isinstance(subdict, dict):
            for _, items in subdict.items():
                if isinstance(items, list):
                    bucket.extend(items)
        flat[level] = bucket
    return flat


def normalize_level_probs(level_probs: Dict[str, float], available_levels: List[str]) -> Tuple[List[str], np.ndarray]:
    """
    Restrict to keys present in available_levels and normalize to sum to 1.
    Raises ValueError if no overlap or invalid totals.
    """
    filtered: List[Tuple[str, float]] = [
        (k, float(v)) for k, v in level_probs.items() if k in available_levels
    ]
    if not filtered:
        raise ValueError("No valid levels found in level_probs intersecting available levels")
    keys, probs = zip(*filtered)
    arr = np.asarray(probs, dtype=float)
    total = arr.sum()
    if total <= 0 or not np.isfinite(total):
        raise ValueError("level_probs must sum to a positive finite number over valid levels")
    arr = arr / total
    return list(keys), arr


def sample_level(keys: List[str], probs: np.ndarray, rng: np.random.Generator) -> str:
    idx = rng.choice(len(keys), p=probs)
    return keys[int(idx)]


def sample_topic(level_topics: Dict[str, List[dict]], level: str, rng: np.random.Generator) -> dict:
    pool = level_topics.get(level, [])
    if not pool:
        raise ValueError(f"No topics available for sampled level: {level}")
    idx = rng.integers(0, len(pool))
    return pool[int(idx)]


