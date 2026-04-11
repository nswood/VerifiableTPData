"""
Data structures and utilities for multi-model quality control grading.

This module provides classes and functions to manage quality control gradings
from multiple models with versioned storage support.
"""

from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Optional, Any


@dataclass
class QualityGrade:
    """Represents a single quality grading from one model run."""

    index: int
    timestamp: str
    problem_quality: int
    problem_quality_comment: str
    solution_completeness: int
    solution_completeness_comment: str
    solution_quality: int
    solution_quality_comment: str
    test_case_quality: int
    test_case_quality_comment: str
    seed_correspondence: Optional[int] = None
    seed_correspondence_comment: Optional[str] = None
    output_seed_correspondence: Optional[int] = None
    output_seed_correspondence_comment: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values."""
        result = asdict(self)
        return {k: v for k, v in result.items() if v is not None}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'QualityGrade':
        """Create QualityGrade from dictionary."""
        # Filter to only known fields
        valid_fields = {
            'index', 'timestamp', 'problem_quality', 'problem_quality_comment',
            'solution_completeness', 'solution_completeness_comment',
            'solution_quality', 'solution_quality_comment',
            'test_case_quality', 'test_case_quality_comment',
            'seed_correspondence', 'seed_correspondence_comment',
            'output_seed_correspondence', 'output_seed_correspondence_comment'
        }
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered_data)


class MultiModelQualityStore:
    """Manages multi-model quality gradings in problem JSON data."""

    QUALITY_GRADINGS_KEY = "quality_gradings"

    def add_grading(
        self,
        data: Dict[str, Any],
        model_id: str,
        scores: Dict[str, Any],
        force_index: Optional[int] = None,
        timestamp: Optional[str] = None
    ) -> int:
        """
        Add a new grading to the problem data.

        Args:
            data: Problem JSON data (will be modified in place)
            model_id: Model identifier (e.g., "gemini-2.5-pro")
            scores: Dictionary of quality scores and comments
            force_index: Force specific index (default: auto-increment)
            timestamp: ISO format timestamp (default: current time)

        Returns:
            The index assigned to this grading
        """
        # Initialize quality_gradings if needed
        if self.QUALITY_GRADINGS_KEY not in data:
            data[self.QUALITY_GRADINGS_KEY] = {}

        # Initialize model list if needed
        if model_id not in data[self.QUALITY_GRADINGS_KEY]:
            data[self.QUALITY_GRADINGS_KEY][model_id] = []

        # Determine index
        model_gradings = data[self.QUALITY_GRADINGS_KEY][model_id]
        if force_index is not None:
            index = force_index
        else:
            # Auto-increment: find max index and add 1
            if model_gradings:
                max_index = max(g.get('index', -1) for g in model_gradings)
                index = max_index + 1
            else:
                index = 0

        # Create timestamp
        if timestamp is None:
            timestamp = datetime.utcnow().isoformat() + 'Z'

        # Build grading dict
        grading_data = {
            'index': index,
            'timestamp': timestamp,
            **scores
        }

        # Add to model's grading list
        model_gradings.append(grading_data)

        return index

    def get_grading(
        self,
        data: Dict[str, Any],
        model_id: str,
        index: int
    ) -> Optional[QualityGrade]:
        """
        Retrieve a specific grading.

        Args:
            data: Problem JSON data
            model_id: Model identifier
            index: Grading index

        Returns:
            QualityGrade if found, None otherwise
        """
        gradings = data.get(self.QUALITY_GRADINGS_KEY, {}).get(model_id, [])

        for grading_dict in gradings:
            if grading_dict.get('index') == index:
                return QualityGrade.from_dict(grading_dict)

        return None

    def get_all_gradings(
        self,
        data: Dict[str, Any],
        model_id: str
    ) -> List[QualityGrade]:
        """
        Get all gradings from one model.

        Args:
            data: Problem JSON data
            model_id: Model identifier

        Returns:
            List of QualityGrade objects, sorted by index
        """
        gradings = data.get(self.QUALITY_GRADINGS_KEY, {}).get(model_id, [])

        quality_grades = [QualityGrade.from_dict(g) for g in gradings]
        quality_grades.sort(key=lambda x: x.index)

        return quality_grades

    def get_latest_grading(
        self,
        data: Dict[str, Any],
        model_id: str
    ) -> Optional[QualityGrade]:
        """
        Get the most recent grading from a model.

        Args:
            data: Problem JSON data
            model_id: Model identifier

        Returns:
            QualityGrade with highest index, None if no gradings exist
        """
        gradings = self.get_all_gradings(data, model_id)

        if not gradings:
            return None

        # Return grading with max index
        return max(gradings, key=lambda x: x.index)

    def get_all_model_ids(self, data: Dict[str, Any]) -> List[str]:
        """
        Get list of all model IDs that have gradings.

        Args:
            data: Problem JSON data

        Returns:
            List of model ID strings
        """
        return list(data.get(self.QUALITY_GRADINGS_KEY, {}).keys())

    def count_gradings(self, data: Dict[str, Any], model_id: str) -> int:
        """
        Count number of gradings for a model.

        Args:
            data: Problem JSON data
            model_id: Model identifier

        Returns:
            Number of gradings
        """
        return len(data.get(self.QUALITY_GRADINGS_KEY, {}).get(model_id, []))


def migrate_legacy_quality(
    data: Dict[str, Any],
    model_id: str = "legacy",
    timestamp: Optional[str] = None
) -> bool:
    """
    Migrate existing 'quality' field to 'quality_gradings' structure.

    Args:
        data: Problem JSON data (will be modified in place)
        model_id: Model ID to assign to migrated grading
        timestamp: Timestamp to use (default: from metadata or "unknown")

    Returns:
        True if migration performed, False if no migration needed
    """
    # Check if migration is needed
    if "quality" not in data:
        return False

    if "quality_gradings" in data and model_id in data["quality_gradings"]:
        # Already migrated
        return False

    # Extract timestamp from metadata if available
    if timestamp is None:
        problem_metadata = data.get("problem_metadata", {})
        timestamp = problem_metadata.get("Date problem was added to the data set", "unknown")

    # Use MultiModelQualityStore to add the grading
    store = MultiModelQualityStore()
    scores = dict(data["quality"])

    store.add_grading(
        data=data,
        model_id=model_id,
        scores=scores,
        force_index=0,
        timestamp=timestamp
    )

    return True
