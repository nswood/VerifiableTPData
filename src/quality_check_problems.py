import argparse
import random
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import tempfile
import re
import sys
import shutil

# Add parent directory to path to import genai from repo root
_script_dir = Path(__file__).parent.absolute()
_repo_root = _script_dir.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

# Import genai from the repo root
try:
    from genai import call_gen_ai, get_api_key  # type: ignore
except ImportError:
    # Fallback if genai module not available
    def get_api_key() -> str:  # type: ignore
        return ""
    def call_gen_ai(prompt: str, api_key: str, model=None) -> str:  # type: ignore
        raise ImportError("genai module not available")

# Import multi-model quality control structures
try:
    from src.qc_data_structures import MultiModelQualityStore
except ImportError:
    # Try alternate import path
    try:
        from qc_data_structures import MultiModelQualityStore
    except ImportError:
        # Fallback - won't support multi-model features
        MultiModelQualityStore = None  # type: ignore

QUALITY_LOG_FILE = Path("quality_report.json")
MAX_SOURCE_CHARS = 4000

_log_lock = threading.Lock()

from tqdm import tqdm


def _format_test_cases(test_data: Optional[Dict]) -> str:
    """
    Format test cases for inclusion in the prompt.
    
    Args:
        test_data: Test data dictionary from problem_details["test"]
    
    Returns:
        Formatted string representation of test cases
    """
    if not test_data:
        return "No test cases provided."
    
    formatted = []
    formatted.append(f"Function Name: {test_data.get('function_name', 'N/A')}")
    formatted.append(f"Arguments: {json.dumps(test_data.get('arguments', []), indent=2)}")
    formatted.append("\nTest Cases:")
    
    test_cases = test_data.get('test_cases', [])
    if not test_cases:
        formatted.append("  No test cases found.")
    else:
        for case in test_cases:
            case_id = case.get('case_id', 'N/A')
            inputs = case.get('inputs', {})
            output = case.get('output', 'N/A')
            output_type = case.get('output_type', 'N/A')
            formatted.append(f"  Case {case_id}:")
            formatted.append(f"    Inputs: {json.dumps(inputs, indent=6)}")
            formatted.append(f"    Expected Output: {output} (type: {output_type})")
    
    tests_passed = test_data.get('tests_passed', None)
    if tests_passed is not None:
        formatted.append(f"\nTests Passed: {tests_passed}")
    
    test_error = test_data.get('test_error', None)
    if test_error:
        formatted.append(f"Test Error: {test_error}")
    
    return "\n".join(formatted)

def _build_prompt(orig: str, problem: str, ans_req: str, solution: str, code: str, test_cases: Optional[str] = None, has_seed: bool = False) -> str:
    """
    Build quality evaluation prompt.

    Args:
        orig: Original source text (only for adapted problems)
        problem: Problem statement
        ans_req: Answer requirements
        solution: Solution text
        code: Code implementation
        test_cases: Formatted test cases string (optional)
        has_seed: Whether this problem was adapted from original source material
    """
    if has_seed:
        return f"""You are evaluating the quality of a physics problem adapted from original source material.

**IMPORTANT: This problem was adapted from seed material to ensure it is verifiable through test case checking. Modifications are expected and should be evaluated based on physics task/topic correspondence, not exact text matching.**

### Original Source Material
{orig}

### Adapted Problem
Problem Statement:
{problem}

Answer Requirements:
{ans_req}

Solution:
{solution}

Code Implementation for Solution:
{code}
{chr(10) + "### Test Cases" + chr(10) + test_cases if test_cases else ""}

### Evaluation
1. Score output-seed correspondence versus the original source. This problem was adapted to ensure verifiability through test case checking (e.g., converting to numeric/categorical outputs, restructuring tasks). Modifications are expected and acceptable. Evaluate whether the output corresponds to the same physics tasks and topics as the original seed. Check for: same physical concepts, same mathematical structure, same problem-solving approach. Differences in format, structure, or presentation style are acceptable if the physics content aligns. If the problem addresses different physics, different problem types, or introduces unrelated concepts, score low. If there is no original source, return 0.
2. Score the quality of the problem: is it well defined and solvable? Base this on the problem statement and answer requirements. Graded according to the distribution below.
3. Score the completeness of the solution. If no solution text is present, return 0. If the solution provides an answer to all problem requirements, return 100. This metric should not consider the quality of the explanation, only whether it addresses all parts of the problem. Graded according to the distribution below.
4. Score the quality of the solution's explanation. Does it provide a clear and correct explanation of the solution? Do NOT take off points for anything related to the python code provided below. The code is simply provided for reference and should not be considered in this score. If the solution is clear, coherent, and shows steps or derivations, return a high score. If the solution lacks clarity or simply states the answer without explanation, return a low score. Graded according to the distribution below.
5. Score the quality of the test cases. Do the test cases reasonably verify if the implementation is correct? Evaluate whether the test cases cover the key aspects of the problem, test edge cases appropriately, and would catch common implementation errors. If test cases are significantly flawed (e.g., they don't test the actual problem requirements, test only trivial cases, have incorrect expected outputs, or would pass incorrect implementations), return 0. If test cases are missing entirely, return 0. If test cases are comprehensive and would correctly verify the solution, return a high score. Graded according to the distribution below.

Scoring system:
- 0–20: Very poor
- 21–40: Poor
- 41–60: Fair
- 61–80: Good
- 81–100: Excellent

Return only a JSON object with exactly these fields. Scores are integers 0–100. Each *_comment is a brief 1–2 sentence explanation (<=200 chars):
{{
  "output_seed_correspondence": <int>,
  "output_seed_correspondence_comment": "<string>",
  "problem_quality": <int>,
  "problem_quality_comment": "<string>",
  "solution_completeness": <int>,
  "solution_completeness_comment": "<string>",
  "solution_quality": <int>,
  "solution_quality_comment": "<string>",
  "test_case_quality": <int>,
  "test_case_quality_comment": "<string>"
}}
"""
    else:
        return f"""You are evaluating the quality of a synthetic physics problem.

### Problem Transcription
The following is a transcription of the problem statement, answer requirements, and solution text written in LaTeX format.
Problem Statement:
{problem}

Answer Requirements:
{ans_req}

Solution:
{solution}

Code Implementation for Solution:
{code}
{chr(10) + "### Test Cases" + chr(10) + test_cases if test_cases else ""}

### Evaluation
1. Score the quality of the problem: is it well defined and solvable? Base this on the problem statement and answer requirements. Graded according to the distribution below. 
2. Score the completeness of the solution. If no solution text is present, return 0. If the solution provides an answer to all problem requirements, return 100. This metric should not consider the quality of the explanation, only whether it addresses all parts of the problem. Graded according to the distribution below.
3. Score the quality of the solution's explanation. Does it provide a clear and correct explanation of the solution? Do NOT take off points for anything related to the python code provided below. The code is simply provided for reference and should not be considered in this score. If the solution is clear, coherent, and shows steps or derivations, return a high score. If the solution lacks clarity or simply states the answer without explanation, return a low score. Graded according to the distribution below.
4. Score the quality of the test cases. Do the test cases reasonably verify if the implementation is correct? Evaluate whether the test cases cover the key aspects of the problem, test edge cases appropriately, and would catch common implementation errors. If test cases are significantly flawed (e.g., they don't test the actual problem requirements, test only trivial cases, have incorrect expected outputs, or would pass incorrect implementations), return 0. If test cases are missing entirely, return 0. If test cases are comprehensive and would correctly verify the solution, return a high score. Graded according to the distribution below.

Scoring system:
- 0–20: Very poor
- 21–40: Poor
- 41–60: Fair
- 61–80: Good
- 81–100: Excellent

Return only a JSON object with exactly these fields. Scores are integers 0–100. Each *_comment is a brief 1–2 sentence explanation (<=200 chars):
{{
  "problem_quality": <int>,
  "problem_quality_comment": "<string>",
  "solution_completeness": <int>,
  "solution_completeness_comment": "<string>",
  "solution_quality": <int>,
  "solution_quality_comment": "<string>",
  "test_case_quality": <int>,
  "test_case_quality_comment": "<string>"
}}
"""


def _build_seed_correspondence_prompt(
    original_seed: Dict[str, Any],
    problem: str,
    ans_req: str,
    solution: str,
    code: str,
    system_prompt: str
) -> str:
    """
    Build a prompt specifically for evaluating seed correspondence.

    Args:
        original_seed: The original_seed dictionary from the problem JSON
        problem: Generated problem statement
        ans_req: Generated answer requirements
        solution: Generated solution
        code: Generated code
        system_prompt: The system prompt used for generation (for reference)

    Returns:
        Formatted prompt for seed correspondence evaluation
    """
    seed_type = original_seed.get("seed_type", "unknown")
    seed_data = original_seed.get("seed_data", {})
    arxiv_id = original_seed.get("arxiv_id", "unknown")

    # Format original seed
    if seed_type == "matched":
        orig_problem = seed_data.get("problem", "")
        orig_solution = seed_data.get("solution", "")
        seed_text = f"Original Problem:\n{orig_problem}\n\nOriginal Solution:\n{orig_solution}"
    elif seed_type == "example":
        orig_example = seed_data.get("example", "")
        seed_text = f"Original Example:\n{orig_example}"
    else:
        seed_text = "No seed data available"

    # Truncate system prompt if too long
    system_prompt_truncated = system_prompt[:2000] + ("..." if len(system_prompt) > 2000 else "")

    return f"""You are evaluating how well a adapted physics problem corresponds to its original seed material.

### Generation Context
The problem below was generated from an original seed using a specific prompt with constraints.
Source: {arxiv_id}
Seed Type: {seed_type}

### Original Seed Material
{seed_text}

### System Prompt Used for Generation
The following constraints were given to the LLM when generating the problem:
{system_prompt_truncated}
... [prompt may be truncated for length]

### Generated Output

Problem Statement:
{problem}

Answer Requirements:
{ans_req}

Solution:
{solution}

Code:
{code}

### Evaluation Task

Evaluate how well the generated problem corresponds to the original seed according to the generation prompt constraints.

**Key Evaluation Criteria:**

Consider the following when evaluating correspondence:
- Does it address the same physics concepts and problem domain?
- Is the physical setting/context preserved?
- Were tasks successfully reformulated to have verifiable outputs (numerical/categorical)?
- Do tasks fit the allowed task types (Direct Calculation, Derivation, Hidden-Coefficient, Ratio/Comparison, Categorical Classification, Logical Check)?
- Does it follow proper structure (Problem, Answer Requirements, Solution, Answer, Code)?

**Important Notes:**
- Modifications to enable verification are EXPECTED and DESIRED
- Format changes, restructuring, and task reformulation are acceptable if physics content aligns
- The goal is "same physics, different format" - not verbatim copying
- If the problem introduces unrelated physics or completely different concepts, score low
- If the seed cannot reasonably be converted to the required format, note this

**Scoring Guidelines:**
- 0-20: Poor correspondence - different physics topics or failed transformation
- 21-40: Fair - related physics but significant deviations or poor task formulation
- 41-60: Moderate - same physics domain but some issues with task transformation
- 61-80: Good - clear correspondence with minor structural issues
- 81-100: Excellent - faithful adaptation with successful task transformation

Return only a JSON object with these fields:
{{
  "seed_correspondence": <int 0-100>,
  "seed_correspondence_comment": "<string, max 300 chars>"
}}
"""


def _grade_seed_correspondence_only(
    json_path: Path,
    api_key: str,
    system_prompt_path: Optional[Path] = None,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Grade only the seed correspondence for a adapted problem.

    Args:
        json_path: Path to the problem JSON file
        api_key: API key for LLM calls
        system_prompt_path: Optional path to the system prompt used for generation

    Returns:
        Dictionary with seed correspondence scores

    Raises:
        ValueError: If problem has no original_seed field or invalid JSON
    """
    try:
        data = _safe_load_json(json_path)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {json_path}: {e}") from e

    # Check if original_seed exists
    if "original_seed" not in data:
        raise ValueError(f"No original_seed field in {json_path}. This function is only for adapted problems.")

    original_seed = data["original_seed"]
    problem = data["problem_details"].get("Problem Statement", "")
    ans_req = data["problem_details"].get("Answer Requirements", "")
    solution = data["problem_details"].get("Solution", "")
    code = data["problem_details"].get("Code", "")

    # Load system prompt if path provided
    system_prompt = ""
    if system_prompt_path and system_prompt_path.exists():
        with open(system_prompt_path, "r", encoding="utf-8") as f:
            system_prompt = f.read()

    # Build prompt
    prompt = _build_seed_correspondence_prompt(
        original_seed=original_seed,
        problem=problem,
        ans_req=ans_req,
        solution=solution,
        code=code,
        system_prompt=system_prompt
    )

    # Call LLM
    allowed_tries = 3
    for attempt in range(allowed_tries):
        try:
            response = call_gen_ai(prompt, api_key, model=model)
            scores = _parse_scores(response)
            break
        except Exception as e:
            print(f"Attempt {attempt+1} failed for {json_path}: {e}")
            if attempt == allowed_tries - 1:
                raise e

    return scores


def _parse_scores(text: str) -> Dict[str, int]:
    """
    Parse JSON scores from API response.
    Handles cases where the response contains extra text or multiple JSON objects.
    """
    # Remove markdown code blocks if present
    response_clean = text.strip()
    if response_clean.startswith('```'):
        lines = response_clean.split('\n')
        if lines[0].startswith('```'):
            lines = lines[1:]
        if lines and lines[-1].strip() == '```':
            lines = lines[:-1]
        response_clean = '\n'.join(lines)
    
    # Find the first { that starts a JSON object
    start_idx = response_clean.find('{')
    if start_idx == -1:
        raise ValueError("No JSON object found in model output")
    
    # Find the matching closing brace by counting braces
    brace_count = 0
    end_idx = start_idx
    for i in range(start_idx, len(response_clean)):
        if response_clean[i] == '{':
            brace_count += 1
        elif response_clean[i] == '}':
            brace_count -= 1
            if brace_count == 0:
                end_idx = i + 1
                break
    
    if brace_count != 0:
        raise ValueError("Incomplete JSON object in model output")
    
    # Extract and parse just the first complete JSON object
    json_str = response_clean[start_idx:end_idx]
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON from model output: {e}")

def _safe_load_json(json_path: Path) -> Dict[str, Any]:
    """
    Safely load JSON from a file, attempting to recover from corruption.
    If the file has extra data after a valid JSON object, extracts just the valid part.
    """
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Try normal parsing first
        return json.loads(content)
    except json.JSONDecodeError as e:
        # If there's extra data, try to extract just the valid JSON part
        if "Extra data" in str(e) or e.pos < len(content):
            # Find the last complete JSON object by counting braces
            brace_count = 0
            last_valid_pos = 0
            for i, char in enumerate(content):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        last_valid_pos = i + 1
            
            if last_valid_pos > 0:
                # Extract just the valid JSON part
                valid_content = content[:last_valid_pos]
                try:
                    data = json.loads(valid_content)
                    # Save the cleaned version back
                    _safe_write_json(json_path, data)
                    print(f"Warning: Recovered corrupted JSON file {json_path} by removing extra data")
                    return data
                except json.JSONDecodeError:
                    pass
        
        # If recovery failed, re-raise the original error
        raise ValueError(f"Failed to parse JSON from {json_path}: {e}") from e
    except Exception as e:
        raise ValueError(f"Failed to read JSON from {json_path}: {e}") from e

def _safe_write_json(json_path: Path, data: Dict[str, Any]) -> None:
    """
    Safely write JSON data to a file using atomic write pattern.
    Writes to a temporary file first, validates it, then atomically replaces the original.
    This prevents file corruption if a write is interrupted.
    """
    # Create temporary file in the same directory
    temp_path = json_path.with_suffix(json_path.suffix + '.tmp')
    try:
        # Write to temporary file
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        
        # Validate the written JSON by reading it back
        with open(temp_path, "r", encoding="utf-8") as f:
            json.load(f)
        
        # Atomically replace the original file
        shutil.move(str(temp_path), str(json_path))
    except Exception as e:
        # Clean up temp file on error
        if temp_path.exists():
            temp_path.unlink()
        raise ValueError(f"Failed to write JSON to {json_path}: {e}") from e

def _log_scores(json_path: Path, scores: Dict[str, int]) -> None:
    with _log_lock:
        try:
            if QUALITY_LOG_FILE.exists():
                log = _safe_load_json(QUALITY_LOG_FILE)
            else:
                log = []
        except Exception as e:
            # If log file is corrupted, start fresh rather than failing
            print(f"Warning: Could not load {QUALITY_LOG_FILE}, starting fresh log: {e}")
            log = []
        
        entry = {"json_path": str(json_path)}
        entry.update(scores)
        log.append(entry)
        _safe_write_json(QUALITY_LOG_FILE, log)

def _quality_check_single(
    json_path: Path,
    api_key: str,
    model: Optional[str] = None,
    qc_model_id: Optional[str] = None,
    force_index: Optional[int] = None,
) -> int:
    try:
        data = _safe_load_json(json_path)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {json_path}: {e}") from e

    problem = data["problem_details"].get("Problem Statement", "")
    ans_req = data["problem_details"].get("Answer Requirements", "")
    solution = data["problem_details"].get("Solution", "")
    code = data["problem_details"].get("Code", "")
    test_data = data["problem_details"].get("test", None)
    test_cases_str = _format_test_cases(test_data) if test_data else None

    # Determine if this is an adapted problem (has original seed material)
    has_seed = "original_seed" in data
    orig = ""
    if has_seed:
        seed = data["original_seed"]
        seed_data = seed.get("seed_data", {})
        orig_problem = seed_data.get("problem", "")
        orig_solution = seed_data.get("solution", "")
        if isinstance(orig_problem, dict):
            orig_problem = orig_problem.get("content", str(orig_problem))
        if isinstance(orig_solution, dict):
            orig_solution = orig_solution.get("content", str(orig_solution))
        orig = f"Problem: {orig_problem}\nSolution: {orig_solution}"

    prompt = _build_prompt(orig, problem, ans_req, solution, code, test_cases=test_cases_str, has_seed=has_seed)
    allowed_tries = 3
    for attempt in range(allowed_tries):
        try:
            response = call_gen_ai(prompt, api_key, model=model)
            scores = _parse_scores(response)
            break
        except Exception as e:
            print(f"Attempt {attempt+1} failed for {json_path}: {e}")
            if attempt == allowed_tries - 1:
                raise e

    # Handle backward compatibility: if old field name exists, rename it
    if "transcription_accuracy" in scores:
        scores["output_seed_correspondence"] = scores.pop("transcription_accuracy")
    if "transcription_accuracy_comment" in scores:
        scores["output_seed_correspondence_comment"] = scores.pop("transcription_accuracy_comment")

    if qc_model_id is None:
        qc_model_id = model if model else "gemini-2.5-pro"

    store = MultiModelQualityStore()
    assigned_index = store.add_grading(
        data=data,
        model_id=qc_model_id,
        scores=scores,
        force_index=force_index
    )

    _safe_write_json(json_path, data)
    _log_scores(json_path, scores)

    return assigned_index


def _quality_check_multi(
    json_path: Path,
    api_key: str,
    model: Optional[str] = None,
    qc_model_id: Optional[str] = None,
    num_runs: int = 1,
    delay_between_runs: float = 0.25
) -> List[int]:
    """
    Perform multiple quality check runs on a single problem.

    Args:
        json_path: Path to problem JSON file
        api_key: API key for grading model
        model: Model to use for grading
        qc_model_id: Model ID for indexing gradings
        num_runs: Number of grading runs to perform
        delay_between_runs: Seconds to wait between API calls

    Returns:
        List of assigned indices for each run
    """
    import time

    try:
        data = _safe_load_json(json_path)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {json_path}: {e}") from e

    problem = data["problem_details"].get("Problem Statement", "")
    ans_req = data["problem_details"].get("Answer Requirements", "")
    solution = data["problem_details"].get("Solution", "")
    code = data["problem_details"].get("Code", "")
    test_data = data["problem_details"].get("test", None)
    test_cases_str = _format_test_cases(test_data) if test_data else None

    has_seed = "original_seed" in data
    orig = ""
    if has_seed:
        seed = data["original_seed"]
        seed_data = seed.get("seed_data", {})
        orig_problem = seed_data.get("problem", "")
        orig_solution = seed_data.get("solution", "")
        if isinstance(orig_problem, dict):
            orig_problem = orig_problem.get("content", str(orig_problem))
        if isinstance(orig_solution, dict):
            orig_solution = orig_solution.get("content", str(orig_solution))
        orig = f"Problem: {orig_problem}\nSolution: {orig_solution}"

    prompt = _build_prompt(orig, problem, ans_req, solution, code, test_cases=test_cases_str, has_seed=has_seed)

    if qc_model_id is None:
        qc_model_id = model if model else "gemini-2.5-pro"

    indices = []
    for run_idx in range(num_runs):
        if model == "test-model":
            scores = {
                "problem_quality": 100,
                "problem_quality_comment": f"Test grading {run_idx + 1}",
                "solution_completeness": 100,
                "solution_completeness_comment": f"Test grading {run_idx + 1}",
                "solution_quality": 100,
                "solution_quality_comment": f"Test grading {run_idx + 1}",
                "test_case_quality": 100,
                "test_case_quality_comment": f"Test grading {run_idx + 1}"
            }
        else:
            allowed_tries = 3
            for attempt in range(allowed_tries):
                try:
                    response = call_gen_ai(prompt, api_key, model=model)
                    scores = _parse_scores(response)
                    break
                except Exception as e:
                    print(f"Attempt {attempt+1} failed for {json_path}: {e}")
                    if attempt == allowed_tries - 1:
                        raise e

        if "transcription_accuracy" in scores:
            scores["output_seed_correspondence"] = scores.pop("transcription_accuracy")
        if "transcription_accuracy_comment" in scores:
            scores["output_seed_correspondence_comment"] = scores.pop("transcription_accuracy_comment")

        store = MultiModelQualityStore()
        assigned_index = store.add_grading(
            data=data,
            model_id=qc_model_id,
            scores=scores,
        )

        indices.append(assigned_index)
        _log_scores(json_path, scores)

        if run_idx < num_runs - 1:
            time.sleep(delay_between_runs)

    _safe_write_json(json_path, data)
    return indices


def _gather_json_files(json_dir: Path) -> List[Path]:
    """Gather top-level JSON files, excluding metadata. Does not recurse into subdirs."""
    all_files = sorted(json_dir.glob("*.json"))
    return [f for f in all_files if f.name not in ("assignment.json", "config.json")]

def main(
    base_dir: Path = Path('.'),
    test_run: bool = False,
    num_workers: int = 24,
    seed_correspondence_only: bool = False,
    system_prompt_path: Optional[Path] = None,
    model: Optional[str] = None,
    qc_model_id: Optional[str] = None,
    qc_runs: int = 1,
    delay_between_calls: float = 2.0,
) -> None:
    base_dir_path = Path(base_dir)
    json_dir = base_dir_path

    api_key = get_api_key(model_name=model)
    json_files = _gather_json_files(json_dir)
    if test_run:
        json_files = random.sample(json_files, min(5, len(json_files)))
    
    if seed_correspondence_only:
        print("Running SEED CORRESPONDENCE ONLY mode")
        # Filter to only process adapted problems with original_seed
        # that are missing seed_correspondence metric
        filtered = []
        for p in json_files:
            try:
                data = _safe_load_json(p)
            except Exception:
                continue
            # Only process if has original_seed and is missing seed_correspondence
            if "original_seed" in data:
                quality = data.get("quality", {})
                if not quality or "seed_correspondence" not in quality:
                    filtered.append(p)
        json_files = filtered
        print(f'Adapted problems to grade for seed correspondence: {len(json_files)}')

        # Process with seed correspondence grading
        def process_single_file(path: Path) -> None:
            try:
                scores = _grade_seed_correspondence_only(path, api_key, system_prompt_path, model=model)

                # Load existing data
                data = _safe_load_json(path)

                # Update or create quality field
                if "quality" not in data:
                    data["quality"] = {}

                # Add seed correspondence scores
                data["quality"]["seed_correspondence"] = scores.get("seed_correspondence", 0)
                data["quality"]["seed_correspondence_comment"] = scores.get("seed_correspondence_comment", "")

                # Save updated JSON
                _safe_write_json(path, data)

                _log_scores(path, scores)
            except Exception as e:
                print(f"Error processing {path}: {e}")

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(process_single_file, path): path for path in json_files}
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Grading seed correspondence"):
                try:
                    fut.result()
                except Exception as e:
                    print(f"Error: {e}")

        return  # Exit early, don't run normal quality check
    
    # Normal quality check mode
    # Determine effective qc_model_id for filtering
    effective_qc_model_id = qc_model_id if qc_model_id else (model if model else "gemini-2.5-pro")

    # Filter files that need QC runs to reach target count
    print(f'Filtering for files that need QC runs to reach {qc_runs} total for model "{effective_qc_model_id}"')
    store = MultiModelQualityStore()
    total_found = len(json_files)
    filtered = []

    for p in json_files:
        try:
            data = _safe_load_json(p)
        except Exception:
            continue
        existing_count = store.count_gradings(data, effective_qc_model_id)
        if existing_count < qc_runs:
            filtered.append(p)

    json_files = filtered
    print(f'Total json files found: {total_found}')
    print(f'Json files to process: {len(json_files)} (need additional QC runs)')

    # Determine how many runs to perform per file
    def process_file(path: Path) -> None:
        try:
            data = _safe_load_json(path)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return

        existing_count = store.count_gradings(data, effective_qc_model_id)
        runs_needed = qc_runs - existing_count
        if runs_needed > 0:
            _quality_check_multi(path, api_key, model, qc_model_id, runs_needed, delay_between_calls)

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_file, path): path for path in json_files}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Processing JSON files"):
            path = futures[fut]
            try:
                fut.result()
            except Exception as e:
                print(f"Error processing {path}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quality check problems")
    parser.add_argument("--base_dir", required=True, type=Path, help="Directory containing problem JSON files")
    parser.add_argument("--test_run", action="store_true")
    parser.add_argument("--num_workers", default=8, type=int, help="Number of parallel workers (default: 8, matching generation pattern)")
    parser.add_argument(
        "--seed_correspondence_only",
        action="store_true",
        help="Only grade seed correspondence for adapted problems"
    )
    parser.add_argument(
        "--system_prompt",
        type=Path,
        default=None,
        help="Path to system prompt used for generation (for context in seed grading)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override the default LLM model used for quality checking",
    )
    parser.add_argument(
        "--qc-model-id",
        type=str,
        default=None,
        help="Model ID for this quality check run (e.g., 'gemini-2.5-pro'). Defaults to --model value if not specified.",
    )
    parser.add_argument(
        "--qc-runs",
        type=int,
        default=1,
        help="Target number of total QC grading runs per problem for the specified model. Will only run additional gradings to reach this total (default: 1)",
    )
    parser.add_argument(
        "--delay-between-calls",
        type=float,
        default=0.25,
        help="Seconds to wait between API calls when running multiple QC runs (default: 3.0). Increase if hitting rate limits.",
    )
    args = parser.parse_args()
    main(
        args.base_dir,
        test_run=args.test_run,
        num_workers=args.num_workers,
        seed_correspondence_only=args.seed_correspondence_only,
        system_prompt_path=args.system_prompt,
        model=args.model,
        qc_model_id=args.qc_model_id,
        qc_runs=args.qc_runs,
        delay_between_calls=args.delay_between_calls,
    )
