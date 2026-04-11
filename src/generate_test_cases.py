import re
import simplejson as json
import os
import shutil
import numpy as np
import ast
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from sympy import Symbol, Function, FunctionClass, sympify, Mul
import argparse
import signal
import sys
import scipy
from concurrent.futures import ProcessPoolExecutor, TimeoutError, as_completed
from tqdm import tqdm

MAX_WORKERS = 24
TIME_LIMIT = 60  # seconds for test execution timeout
REPAIR_TIMEOUT = 300  # seconds for repair process timeout (5 minutes)

# Global variable for base directory (set in main())
BASE_DIR = Path(__file__).resolve().parent.parent

# Ensure parent directory is on the import path for local genai module
sys.path.insert(0, str(BASE_DIR))

def ensure_test_log_file_exists(base_dir: Path) -> Path:
    """
    Ensure the test cases log directory and file exist.
    
    Args:
        base_dir: Base directory for the project
        
    Returns:
        Path to the test cases log file
    """
    log_dir = base_dir / "file_logs"
    log_dir.mkdir(exist_ok=True)
    
    log_file = log_dir / "test_cases_attempted.json"
    if not log_file.exists():
        # Create file with empty list
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump([], f, indent=2)
    
    return log_file

try:
    from genai import call_gen_ai, read_api_key  # type: ignore
    GENAI_AVAILABLE = True
except ImportError:
    # Fallback if genai module not found - AI generation won't work but rest of code will
    GENAI_AVAILABLE = False
    def call_gen_ai(*args, **kwargs):  # type: ignore
        raise ImportError("genai module not available")
    def read_api_key(*args, **kwargs):  # type: ignore
        raise ImportError("genai module not available")

# Define ProductReducedFunction class
class ProductReducedFunction(Function):
    @classmethod
    def eval(cls, *args):
        args = tuple(map(sympify, args))
        multiplied_args = Mul(*args)
        if args[0] != multiplied_args:
            return cls(multiplied_args)


def extract_from_llm(code_str: str) -> str:
    """
    Extracts Python code from between ```python and ``` by splitting.
    Returns the inner code, or '' if no matching fences are found.
    """
    # Split off everything before the first ```python
    parts = code_str.split('```python', 1)
    if len(parts) < 2:
        return ''
    # Now split off everything after the closing ```
    code_and_rest = parts[1]
    code_parts = code_and_rest.split('```', 1)
    # Take just the code portion, and strip any leading newlines
    return code_parts[0].lstrip('\r\n')

def parse_function_signature(code_str: str) -> Tuple[str, List[Tuple[str, str]], str]:
    """Parse function signature to get name, args and return type"""
    code_str = extract_python_code(code_str)
    
    func_match = re.search(
        r"def\s+(\w+)\s*\(([^)]*)\)(?:\s*->\s*([^:]+))?\s*:", 
        code_str
    )
    if not func_match:
        raise ValueError("No valid function definition found")
        
    func_name = func_match.group(1)
    arg_string = func_match.group(2).strip()
    return_type = func_match.group(3).strip() if func_match.group(3) else 'None'
    
    args = []
    if arg_string:
        arg_pattern = re.compile(r"(\w+)\s*:\s*([\w\.\[\]]+)")
        args = arg_pattern.findall(arg_string)
        # Normalize Optional[type] to just type for test generation
        normalized_args = []
        for name, type_ in args:
            # Handle Optional[type] -> extract inner type
            if type_.startswith('Optional[') and type_.endswith(']'):
                inner_type = type_[9:-1]  # Remove "Optional[" and "]"
                normalized_args.append((name, inner_type))
            else:
                normalized_args.append((name, type_))
        args = normalized_args
    
    return func_name, args, return_type
def extract_python_code(text: str) -> str:
    """
    Return the substring between \\begin{python} and \\end{python} in `text`.
    If no such tags are found, returns an empty string.
    Removes LaTeX commands (like \\subsection, \\section, etc.) from the extracted code.
    """
    pattern = r"\\begin\{python\}(.*?)\\end\{python\}"
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        return ""
    
    code = match.group(1).strip()
    
    # Remove LaTeX commands that might appear in the code block
    # Common LaTeX commands: \subsection, \section, \subsubsection, \paragraph, etc.
    # Pattern matches: \command{...} or \command[...]{...}
    latex_command_pattern = r'\\[a-zA-Z]+\*?(?:\[[^\]]*\])?(?:\{[^\}]*\})?'
    
    # Remove LaTeX commands line by line
    lines = code.split('\n')
    cleaned_lines = []
    for line in lines:
        # Remove LaTeX commands from the line
        cleaned_line = re.sub(latex_command_pattern, '', line)
        # Only keep the line if it's not empty or just whitespace after cleaning
        if cleaned_line.strip():
            cleaned_lines.append(cleaned_line)
        # If the original line had content but was just a LaTeX command, skip it
        elif not line.strip().startswith('\\'):
            # Keep lines that don't start with backslash (might be empty lines we want to preserve)
            cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines).strip()

def extract_valid_string_keys(code_str: str, arg_name: str, docstring: str = "") -> Optional[List[str]]:
    """
    Extract valid string keys from code by detecting dictionary lookups and if/elif chains.
    
    Args:
        code_str: The Python code string (may include LaTeX python blocks)
        arg_name: Name of the argument to find valid keys for
        docstring: Optional docstring (not used here, but kept for consistency)
        
    Returns:
        List of valid string keys if found, None otherwise
    """
    # Extract Python code from LaTeX blocks if needed
    python_code = extract_python_code(code_str)
    if not python_code:
        python_code = code_str
    
    valid_keys = []
    
    # Pattern 1: Dictionary definition followed by .get(arg_name) or [arg_name]
    # Look for: dict_name = {"key1": value1, "key2": value2, ...}
    # Then: dict_name.get(arg_name) or dict_name[arg_name]
    dict_pattern = r'(\w+)\s*=\s*\{([^}]*)\}'
    for match in re.finditer(dict_pattern, python_code):
        dict_name = match.group(1)
        dict_content = match.group(2)
        
        # Check if this dict is used with the argument
        # Look for dict_name.get(arg_name) or dict_name[arg_name]
        usage_pattern = rf'{re.escape(dict_name)}\s*\.\s*get\s*\(\s*{re.escape(arg_name)}|{re.escape(dict_name)}\s*\[\s*{re.escape(arg_name)}'
        if re.search(usage_pattern, python_code):
            # Extract string keys from dictionary
            # Match "key" or 'key' patterns
            key_pattern = r'["\']([^"\']+)["\']\s*:'
            keys = re.findall(key_pattern, dict_content)
            if keys:
                valid_keys.extend(keys)
    
    # Pattern 2: if/elif chains with string comparisons
    # Look for: if arg_name == "value1": or elif arg_name == "value2":
    if_pattern = rf'if\s+{re.escape(arg_name)}\s*==\s*["\']([^"\']+)["\']'
    elif_pattern = rf'elif\s+{re.escape(arg_name)}\s*==\s*["\']([^"\']+)["\']'
    
    if_matches = re.findall(if_pattern, python_code)
    elif_matches = re.findall(elif_pattern, python_code)
    
    if if_matches or elif_matches:
        valid_keys.extend(if_matches)
        valid_keys.extend(elif_matches)
    
    # Remove duplicates while preserving order
    if valid_keys:
        seen = set()
        unique_keys = []
        for key in valid_keys:
            if key not in seen:
                seen.add(key)
                unique_keys.append(key)
        return unique_keys
    
    return None

def extract_valid_values_from_docstring(docstring: str, arg_name: str) -> Optional[List[str]]:
    """
    Extract valid string values from docstring patterns.
    
    Args:
        docstring: The docstring text
        arg_name: Name of the argument to find valid values for
        
    Returns:
        List of valid string values if found, None otherwise
    """
    if not docstring:
        return None
    
    valid_values = []
    
    # Pattern 1: "One of {"A", "B", "C", "D"}"
    pattern1 = r'One of\s*\{["\']([^"\']+)["\'](?:\s*,\s*["\']([^"\']+)["\'])*\}'
    match1 = re.search(pattern1, docstring, re.IGNORECASE)
    if match1:
        # Extract all quoted strings from the match
        all_quoted = re.findall(r'["\']([^"\']+)["\']', match1.group(0))
        if all_quoted:
            valid_values.extend(all_quoted)
    
    # Pattern 2: "Allowed values: "value1", "value2", "value3""
    pattern2 = rf'{re.escape(arg_name)}[^:]*:\s*["\']([^"\']+)["\'](?:\s*,\s*["\']([^"\']+)["\'])*'
    match2 = re.search(pattern2, docstring, re.IGNORECASE)
    if match2:
        all_quoted = re.findall(r'["\']([^"\']+)["\']', match2.group(0))
        if all_quoted:
            valid_values.extend(all_quoted)
    
    # Pattern 3: "Can be one of "value1", "value2", "value3""
    pattern3 = r'Can be one of\s*["\']([^"\']+)["\'](?:\s*,\s*["\']([^"\']+)["\'])*'
    match3 = re.search(pattern3, docstring, re.IGNORECASE)
    if match3:
        all_quoted = re.findall(r'["\']([^"\']+)["\']', match3.group(0))
        if all_quoted:
            valid_values.extend(all_quoted)
    
    # Pattern 4: Look for arg_name description with explicit list
    # e.g., "arg_name: One of {"A", "B", "C", "D"}"
    escaped_arg_name = re.escape(arg_name)
    pattern4 = escaped_arg_name + r'[^:]*:\s*[^.]*One of\s*\{["\']([^"\']+)["\'](?:\s*,\s*["\']([^"\']+)["\'])*\}'
    match4 = re.search(pattern4, docstring, re.IGNORECASE)
    if match4:
        all_quoted = re.findall(r'["\']([^"\']+)["\']', match4.group(0))
        if all_quoted:
            valid_values.extend(all_quoted)
    
    # Remove duplicates while preserving order
    if valid_values:
        seen = set()
        unique_values = []
        for val in valid_values:
            if val not in seen:
                seen.add(val)
                unique_values.append(val)
        return unique_values
    
    return None

def extract_docstring_from_code(code_str: str) -> str:
    """
    Extract docstring from Python code.
    
    Args:
        code_str: The Python code string (may include LaTeX python blocks)
        
    Returns:
        The docstring content, or empty string if not found
    """
    python_code = extract_python_code(code_str)
    if not python_code:
        python_code = code_str
    
    # Look for triple-quoted docstrings (both """ and ''')
    patterns = [
        r'"""(.*?)"""',  # Triple double quotes
        r"'''(.*?)'''",   # Triple single quotes
    ]
    
    for pattern in patterns:
        match = re.search(pattern, python_code, re.DOTALL)
        if match:
            return match.group(1).strip()
    
    return ""

def extract_tuple_size_from_docstring(docstring: str, arg_name: str) -> Optional[int]:
    """
    Extract tuple size hint from docstring.
    
    Args:
        docstring: The docstring text
        arg_name: Name of the argument
        
    Returns:
        Tuple size if found (e.g., 4 for "4-element tuples"), None otherwise
    """
    if not docstring:
        return None
    
    # Look for patterns like "4-element tuples" or "tuple of 3 elements"
    # Also check for arg_name-specific mentions
    patterns = [
        rf'{re.escape(arg_name)}[^.]*?(\d+)[-\s]*element[^\s]*\s*tuple',
        rf'{re.escape(arg_name)}[^.]*?tuple[^.]*?(\d+)[-\s]*element',
        rf'tuple[^.]*?(\d+)[-\s]*element',
        rf'(\d+)[-\s]*element[^\s]*\s*tuple',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, docstring, re.IGNORECASE)
        if match:
            try:
                return int(match.group(1))
            except (ValueError, IndexError):
                continue
    
    return None

def analyze_unsafe_operations(code_str: str, args: List[Tuple[str, str]]) -> Dict[str, List[str]]:
    """
    Analyze code to identify potential unsafe operations on arguments.
    
    Args:
        code_str: Python code string
        args: List of (name, type) tuples for function arguments
        
    Returns:
        Dictionary mapping argument names to list of risk types
    """
    risks = {name: [] for name, _ in args}
    arg_names = set(name for name, _ in args)
    
    try:
        # Clean code wrapper
        python_code = extract_python_code(code_str)
        if not python_code:
            python_code = code_str
            
        tree = ast.parse(python_code)
        
        for node in ast.walk(tree):
            # Check for division
            if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Div, ast.FloorDiv)):
                # Check if denominator involves an argument
                for subnode in ast.walk(node.right):
                    if isinstance(subnode, ast.Name) and subnode.id in arg_names:
                        if 'division_by_zero' not in risks[subnode.id]:
                            risks[subnode.id].append('division_by_zero')
                            
            # Check for function calls (factorial, sqrt, log)
            if isinstance(node, ast.Call):
                func_name = ""
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    func_name = node.func.attr
                    
                if func_name in ['factorial']:
                    # check arguments
                    for arg in node.args:
                        for subnode in ast.walk(arg):
                            if isinstance(subnode, ast.Name) and subnode.id in arg_names:
                                if 'negative_factorial' not in risks[subnode.id]:
                                    risks[subnode.id].append('negative_factorial')
                                    
                if func_name in ['sqrt', 'log', 'log10', 'log2']:
                     for arg in node.args:
                        for subnode in ast.walk(arg):
                            if isinstance(subnode, ast.Name) and subnode.id in arg_names:
                                if 'domain_error' not in risks[subnode.id]:
                                    risks[subnode.id].append('domain_error')
                                    
    except (SyntaxError, ValueError):
        pass
        
    return risks

def is_input_safe(inputs: Dict[str, Any], unsafe_ops: Dict[str, List[str]], enum_values: Dict[str, List[str]]) -> bool:
    """
    Check if inputs are safe based on identified risks.
    
    Args:
        inputs: Dictionary of input values
        unsafe_ops: Dictionary of risks per argument
        enum_values: Dictionary of valid enum values per argument
        
    Returns:
        True if inputs are considered safe, False otherwise
    """
    for arg_name, value in inputs.items():
        # Check enum constraints first
        if arg_name in enum_values:
            # If it's an enum argument, the value MUST be in the valid set
            # unless the valid set is empty (failed extraction)
            valid_set = enum_values[arg_name]
            if valid_set and str(value) not in valid_set:
                return False
                
        # Check identified risks
        if arg_name in unsafe_ops:
            risks = unsafe_ops[arg_name]
            
            if 'division_by_zero' in risks:
                # Check for zero or near-zero values
                try:
                    if isinstance(value, (int, float, complex)):
                        if abs(value) < 1e-9:
                            return False
                except (ValueError, TypeError):
                    pass
                    
            if 'negative_factorial' in risks:
                # Check for negative integers
                try:
                    if isinstance(value, (int, float)) and value < 0:
                        return False
                except (ValueError, TypeError):
                    pass
                    
            if 'domain_error' in risks:
                # Check for domain errors (negative sqrt, non-positive log)
                try:
                    # For complex numbers, domain errors are less common (cmath handles them)
                    # But for pure float inputs intended for math.sqrt/log, negative is bad
                    if isinstance(value, (int, float)):
                        if value <= 0: # Conservative for log
                             return False
                except (ValueError, TypeError):
                    pass
                    
    return True

def generate_test_value(arg_type: str, arg_name: str = "", code_str: str = "", docstring: str = "") -> Any:
    """
    Generate a single test value based on argument type.
    
    Args:
        arg_type: Type of the argument
        arg_name: Name of the argument (for extracting valid keys)
        code_str: Python code string (for extracting valid keys from code)
        docstring: Docstring (for extracting valid values from documentation)
        
    Returns:
        Generated test value
    """
    # For string types, try to extract valid keys from code or docstring
    if arg_type == 'str' and arg_name:
        # First try extracting from code (most reliable)
        valid_keys = extract_valid_string_keys(code_str, arg_name, docstring)
        
        # If not found in code, try docstring
        if not valid_keys:
            valid_keys = extract_valid_values_from_docstring(docstring, arg_name)
        
        # If valid keys found, randomly sample from them
        if valid_keys:
            return np.random.choice(valid_keys)
    
    # For tuple types, try to extract size from docstring
    if arg_type == 'tuple' and arg_name:
        tuple_size = extract_tuple_size_from_docstring(docstring, arg_name)
        if tuple_size:
            # Generate tuple of specified size with random numbers
            return tuple(np.random.uniform(-10.0, 10.0, size=tuple_size).tolist())
        else:
            # Default: generate a 2-5 element tuple with random numbers
            size = np.random.randint(2, 6)
            return tuple(np.random.uniform(-10.0, 10.0, size=size).tolist())
    
    # Special case: Check if this is a spacetime tuple that needs varied separations
    # This is detected in create_test_cases, not here
    
    # For complex types, generate complex numbers
    if arg_type == 'complex':
        real_part = np.random.uniform(-10.0, 10.0)
        imag_part = np.random.uniform(-10.0, 10.0)
        return complex(real_part, imag_part)
    
    # Fall back to original generators for other types or when no valid keys found
    generators = {
        'float': lambda: np.random.uniform(0.1, 10.0),
        'int': lambda: np.random.randint(-10, 10),
        'str': lambda: ''.join(np.random.choice(list('abcdefghijklmnopqrstuvwxyz'), 
                                              np.random.randint(3, 8))),
        'bool': lambda: bool(np.random.randint(0, 2)),
        'np.ndarray': lambda: np.random.uniform(-10, 10, 
                                              size=np.random.randint(2, 6)).tolist(),
        'Symbol': lambda: np.random.choice(list('abcdefghijklmnopqrstuvwxyz')),
        'FunctionClass': lambda: "ProductReducedFunction"
    }
    
    return generators.get(arg_type, lambda: None)()

def make_json_serializable(obj: Any) -> Any:
    """
    Convert non-JSON-serializable objects to JSON-serializable formats.
    
    Args:
        obj: Object to convert
        
    Returns:
        JSON-serializable version of the object
    """
    if isinstance(obj, complex):
        # Convert complex numbers to string format (e.g., "(1+2j)")
        return str(obj)
    elif isinstance(obj, dict):
        return {key: make_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    else:
        return obj

def build_gemini_test_case_prompt(
    problem_statement: str,
    solution: str,
    code_str: str,
    func_name: str,
    args: List[Tuple[str, str]],
    docstring: str = ""
) -> str:
    """
    Build a prompt for Gemini to generate diverse test cases.
    
    Args:
        problem_statement: The problem statement text
        solution: The solution explanation
        code_str: The function code
        func_name: Name of the function
        args: List of (name, type) tuples for function arguments
        docstring: Function docstring
        
    Returns:
        Formatted prompt string for Gemini
    """
    args_description = ", ".join([f"{name}: {type_}" for name, type_ in args])
    
    prompt = f"""You are generating test cases for a physics problem. Your task is to create test cases that comprehensively cover all possible outcomes of the function.

PROBLEM STATEMENT:
{problem_statement}

SOLUTION:
{solution}

FUNCTION SIGNATURE:
def {func_name}({args_description}) -> ...

FUNCTION CODE:
{code_str}

FUNCTION DOCSTRING:
{docstring}

TASK:
Generate test cases for this function following these rules:
1. MINIMUM: Generate at least 1 test case
2. MAXIMUM: Generate up to 10 test cases
3. SPECIAL CASE: If the function has no input parameters, you may generate just 1 test case
4. COVERAGE: If the function has varying outputs, generate enough test cases to cover ALL possible distinct outcomes
   - You may only generate test cases for tasks which are mentioned in the problem statement and solution.
   - Do not generate test cases for tasks which are not mentioned in the problem statement and solution.
   - If there are 1-10 distinct outcomes, generate one test case for each outcome
   - If there are more than 10 distinct outcomes, generate 10 diverse test cases that cover the most important outcomes
   - If there is only one possible outcome (or the function has no inputs), you may generate just 1 test case

The test cases should:
- The GOAL is to verify the PHYSICS REASONING, not to test error handling, input validation, or boundary conditions.
- Cover different scenarios mentioned in the problem and solution 
- Focus on testing physics logic with valid, physically meaningful inputs
- Avoid edge cases that test implementation robustness (e.g., division by zero, negative factorials, invalid enum keys, physically invalid values)
- Use valid input types and formats
- Be diverse - avoid repetitive or similar test cases
- Ensure each distinct outcome is represented at least once (up to 10 total cases)

For each test case, provide the input values for each argument. Pay special attention to:
- Enum-like string parameters: use ALL valid values specified in the code/docstring to ensure coverage. DO NOT generate invalid keys to test error handling.
- Zero values: Only include zero if it's physically meaningful and valid for the formula. Do not use zero to test division-by-zero error handling.
- Boundary values: Include physically valid boundary values (e.g., m=0 for massless particles), but avoid values that cause math domain errors.
- Complex numbers: if applicable, include various combinations of real/imaginary parts
- Tuples: ensure proper size and meaningful values

OUTPUT FORMAT:
Return a JSON object with this exact structure:
{{
    "test_cases": [
        {{
            "case_id": 1,
            "inputs": {{
                "arg_name_1": value1,
                "arg_name_2": value2,
                ...
            }}
        }},
        ...
    ]
}}

IMPORTANT:
- Generate between 1 and 10 test cases (inclusive)
- If the function has no input parameters, 1 test case is acceptable
- If there are varying outputs, generate enough test cases to cover all distinct outcomes (up to 10)
- Prioritize covering all distinct outcomes over quantity
- Use the exact argument names from the function signature: {", ".join([name for name, _ in args]) if args else "(no arguments)"}
- For complex numbers, use string format like "(1+2j)" or [real, imag] format
- For tuples, use list format like [1, 2, 3, 4]
- Ensure all values are valid for their types
- Make test cases diverse and meaningful

Return ONLY the JSON object, no additional text or markdown formatting."""
    
    return prompt


def build_repair_prompt(
    problem_statement: str,
    solution: str,
    code_str: str,
    func_name: str,
    args: List[Tuple[str, str]],
    existing_test_cases: Dict,
    review_comments: List[Dict],
    docstring: str = ""
) -> str:
    """
    Build a prompt for Gemini to repair test cases based on QC feedback.
    
    Args:
        problem_statement: The problem statement text
        solution: The solution explanation
        code_str: The function code
        func_name: Name of the function
        args: List of (name, type) tuples for function arguments
        existing_test_cases: Current test cases dictionary
        review_comments: List of QC review comments about test case quality
        docstring: Function docstring
        
    Returns:
        Formatted prompt string for Gemini to repair test cases
    """
    args_description = ", ".join([f"{name}: {type_}" for name, type_ in args])
    
    # Format existing test cases
    existing_tests_str = ""
    if existing_test_cases and "test_cases" in existing_test_cases:
        for case in existing_test_cases["test_cases"]:
            existing_tests_str += f"\nCase {case.get('case_id', '?')}:\n"
            existing_tests_str += f"  Inputs: {json.dumps(case.get('inputs', {}))}\n"
            if "output" in case:
                existing_tests_str += f"  Output: {case['output']}\n"
            if "output_type" in case:
                existing_tests_str += f"  Output Type: {case['output_type']}\n"
    
    # Format review comments
    reviews_str = ""
    for review in review_comments:
        model_id = review.get("model_id", "unknown")
        index = review.get("index", 0)
        score = review.get("test_case_quality", "N/A")
        comment = review.get("test_case_quality_comment", "No comment provided")
        reviews_str += f"\n- {model_id} (review {index}, score: {score}):\n  \"{comment}\"\n"
    
    prompt = f"""You are repairing test cases for a physics problem based on quality control feedback. The current test cases have issues that need to be fixed.

PROBLEM STATEMENT:
{problem_statement}

SOLUTION:
{solution}

FUNCTION SIGNATURE:
def {func_name}({args_description}) -> ...

FUNCTION CODE:
{code_str}

FUNCTION DOCSTRING:
{docstring}

CURRENT TEST CASES (with issues):
{existing_tests_str if existing_tests_str else "No existing test cases"}

QUALITY CONTROL REVIEW FEEDBACK:
{reviews_str if reviews_str else "No review comments available"}

YOUR TASK:
Based on the above QC feedback, generate IMPROVED test cases that address the identified issues. Key points:

1. READ THE FEEDBACK CAREFULLY - The reviewers identified specific problems:
   - Incorrect expected values
   - Numerical precision issues
   - Test cases that contradict physics principles
   - Boolean/flag values that should be different
   
2. FIX THE IDENTIFIED ISSUES:
   - If reviewers say expected values are wrong, compute correct ones
   - If reviewers mention numerical tolerance issues, use appropriate precision
   - If reviewers say physics is contradicted (e.g., unitarity checks), fix the expected outcomes
   - If reviewers mention fragile float expectations, ensure values are correct

3. GENERATE 1-10 TEST CASES that:
   - Cover the physics correctly
   - Have accurate expected behavior
   - Test diverse parameter regimes
   - Are consistent with the problem's physics (e.g., conservation laws, unitarity)

4. IMPORTANT PHYSICS CONSIDERATIONS:
   - If the problem involves unitarity/probability conservation, test cases should reflect that
   - Use physically meaningful parameter values
   - Expected outputs should match what the code actually produces for correct physics

OUTPUT FORMAT:
Return a JSON object with this exact structure:
{{
    "test_cases": [
        {{
            "case_id": 1,
            "inputs": {{
                "arg_name_1": value1,
                "arg_name_2": value2,
                ...
            }}
        }},
        ...
    ]
}}

IMPORTANT:
- Generate between 1 and 10 test cases
- Use the exact argument names from the function signature: {", ".join([name for name, _ in args]) if args else "(no arguments)"}
- For complex numbers, use string format like "(1+2j)" or [real, imag] format
- For tuples, use list format like [1, 2, 3, 4]
- Ensure all values are valid for their types
- Make test cases that will PASS when run against correct physics implementation

Return ONLY the JSON object, no additional text or markdown formatting."""
    
    return prompt


def parse_gemini_test_cases(
    gemini_response: str,
    func_name: str,
    args: List[Tuple[str, str]]
) -> Optional[Dict]:
    """
    Parse test cases from Gemini's response.
    
    Args:
        gemini_response: Raw response from Gemini API
        func_name: Name of the function
        args: List of (name, type) tuples for function arguments
        
    Returns:
        Dictionary in standard test case format, or None if parsing fails
    """
    try:
        # Try to extract JSON from the response
        # Remove markdown code blocks if present
        response_clean = gemini_response.strip()
        
        # Remove ```json or ``` markers
        if response_clean.startswith('```'):
            lines = response_clean.split('\n')
            # Remove first line (```json or ```)
            if lines[0].startswith('```'):
                lines = lines[1:]
            # Remove last line if it's ```
            if lines and lines[-1].strip() == '```':
                lines = lines[:-1]
            response_clean = '\n'.join(lines)
        
        # Try to find JSON object - look for opening brace and try to parse
        # Find the first { that starts a JSON object
        start_idx = response_clean.find('{')
        if start_idx != -1:
            # Try to find the matching closing brace by counting braces
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
            if end_idx > start_idx:
                response_clean = response_clean[start_idx:end_idx]
        
        # Parse JSON
        parsed = json.loads(response_clean)
        
        if "test_cases" not in parsed:
            return None
        
        # Convert to standard format
        test_cases = {
            "function_name": func_name,
            "arguments": [{"name": name, "type": type_} for name, type_ in args],
            "test_cases": []
        }
        
        # Process each test case
        for idx, case in enumerate(parsed["test_cases"], 1):
            inputs = case.get("inputs", {})
            
            # Validate inputs match function arguments
            valid_inputs = {}
            for arg_name, arg_type in args:
                if arg_name in inputs:
                    value = inputs[arg_name]
                    # Convert types if needed
                    value = convert_gemini_value_to_type(value, arg_type)
                    valid_inputs[arg_name] = value
                else:
                    # Missing argument - skip this test case or use default?
                    return None  # Strict validation - all args must be present
            
            # Make JSON serializable
            valid_inputs = make_json_serializable(valid_inputs)
            
            test_cases["test_cases"].append({
                "case_id": idx,
                "inputs": valid_inputs
            })
        
        if len(test_cases["test_cases"]) == 0:
            return None
        
        return test_cases
    
    except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
        # Parsing failed
        return None

def convert_gemini_value_to_type(value: Any, arg_type: str) -> Any:
    """
    Convert a value from Gemini's response to the correct Python type.
    
    Args:
        value: Value from Gemini (might be string, list, etc.)
        arg_type: Expected type
        
    Returns:
        Converted value of correct type
    """
    # Handle complex numbers (string format like "(1+2j)" or list [real, imag])
    if arg_type == 'complex':
        if isinstance(value, str):
            # Remove parentheses and convert
            value_clean = value.strip('()')
            try:
                return complex(value_clean)
            except ValueError:
                pass
        elif isinstance(value, list) and len(value) == 2:
            return complex(float(value[0]), float(value[1]))
        return value
    
    # Handle tuples (might come as lists)
    if arg_type == 'tuple':
        if isinstance(value, list):
            return tuple(value)
        return value
    
    # Handle floats (might come as strings)
    if arg_type == 'float':
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                pass
        return value
    
    # Handle ints (might come as strings)
    if arg_type == 'int':
        if isinstance(value, str):
            try:
                return int(value)
            except ValueError:
                pass
        return value
    
    # Handle bools (might come as strings)
    if arg_type == 'bool':
        if isinstance(value, str):
            return value.lower() in ['true', '1', 'yes']
        return value
    
    return value

def generate_test_cases_with_gemini(
    problem_statement: str,
    solution: str,
    code_str: str,
    func_name: str,
    args: List[Tuple[str, str]],
    docstring: str = "",
    api_key: str = "",
    num_cases: int = 10
) -> Optional[Dict]:
    """
    Generate test cases using Gemini API.
    
    The AI will generate between 1 and 10 test cases, prioritizing coverage of all
    distinct outcomes. If the function has no inputs, it may generate just 1 test case.
    If there are 1-10 distinct outcomes, it generates one test case per outcome.
    If there are more than 10, it generates 10 diverse cases covering the most important outcomes.
    
    Args:
        problem_statement: The problem statement text
        solution: The solution explanation
        code_str: The function code
        func_name: Name of the function
        args: List of (name, type) tuples for function arguments
        docstring: Function docstring
        api_key: Gemini API key
        num_cases: Maximum number of test cases (default 10, but AI will determine
                   actual count based on coverage needs, between 1-10)
        
    Returns:
        Dictionary containing test cases in standard format, or None if generation fails
    """
    if not GENAI_AVAILABLE:
        return None
    
    if not api_key:
        try:
            api_key = read_api_key()
        except ImportError:
            return None
    
    try:
        # Build prompt
        prompt = build_gemini_test_case_prompt(
            problem_statement=problem_statement,
            solution=solution,
            code_str=code_str,
            func_name=func_name,
            args=args,
            docstring=docstring
        )
        
        # Call Gemini API
        response = call_gen_ai(prompt, api_key)
        
        # Parse response
        test_cases = parse_gemini_test_cases(response, func_name, args)
        
        if test_cases:
            # Validate test cases against unsafe operations
            unsafe_ops = analyze_unsafe_operations(code_str, args)
            
            # Pre-extract valid enum values for validation
            enum_values = {}
            for name, type_ in args:
                if type_ == 'str':
                    valid_keys = extract_valid_string_keys(code_str, name, docstring)
                    if not valid_keys:
                        valid_keys = extract_valid_values_from_docstring(docstring, name)
                    if valid_keys:
                        enum_values[name] = valid_keys
            
            # Filter invalid test cases
            valid_cases = []
            for case in test_cases["test_cases"]:
                inputs = case.get("inputs", {})
                if is_input_safe(inputs, unsafe_ops, enum_values):
                    valid_cases.append(case)
            
            # Update test cases list
            test_cases["test_cases"] = valid_cases
            
            if not valid_cases:
                return None
        
        return test_cases
    
    except Exception as e:
        # Any error - return None to trigger fallback
        return None

def create_test_cases(func_name: str, args: List[Tuple[str, str]], 
                     code_str: str = "", docstring: str = "", num_cases: int = 5) -> Dict:
    """
    Generate test cases for a function.
    
    Args:
        func_name: Name of the function
        args: List of (name, type) tuples for function arguments
        code_str: Python code string (for extracting valid keys)
        docstring: Docstring (for extracting valid values)
        num_cases: Number of test cases to generate
        
    Returns:
        Dictionary containing test cases
    """
    # Extract docstring from code if not provided separately
    if not docstring and code_str:
        docstring = extract_docstring_from_code(code_str)
    
    test_cases = {
        "function_name": func_name,
        "arguments": [{"name": name, "type": type_} for name, type_ in args],
        "test_cases": []
    }
    
    # Pre-extract valid enum values for each string argument
    enum_values = {}
    for name, type_ in args:
        if type_ == 'str':
            # Try to extract valid keys
            valid_keys = extract_valid_string_keys(code_str, name, docstring)
            if not valid_keys:
                valid_keys = extract_valid_values_from_docstring(docstring, name)
            if valid_keys:
                enum_values[name] = valid_keys
    
    # Check if any arguments might need zero values (for == 0 or != 0 checks)
    args_needing_zero = {}
    # Check if function uses spacetime intervals (for tuple arguments)
    needs_spacetime_variety = False
    spacetime_tuple_args = []
    
    if code_str:
        python_code = extract_python_code(code_str)
        if python_code:
            # Check for spacetime interval calculations (interval_sq, spacelike, timelike)
            if re.search(r'interval_sq|spacelike|timelike|lightlike', python_code, re.IGNORECASE):
                # Find tuple arguments that might be spacetime coordinates
                for name, type_ in args:
                    if type_ == 'tuple':
                        # Check if this tuple is used in interval calculation
                        if re.search(rf'{re.escape(name)}', python_code):
                            spacetime_tuple_args.append(name)
                            needs_spacetime_variety = True
            
            for name, type_ in args:
                if type_ in ['complex', 'float', 'int']:
                    # Check if function compares this argument to zero
                    zero_patterns = [
                        rf'{re.escape(name)}\s*==\s*0',
                        rf'{re.escape(name)}\s*!=\s*0',
                        rf'{re.escape(name)}\s*==\s*0\.0',
                        rf'{re.escape(name)}\s*!=\s*0\.0',
                    ]
                    for pattern in zero_patterns:
                        if re.search(pattern, python_code):
                            args_needing_zero[name] = type_
                            break
    
    # Generate test cases
    used_enum_values = {name: set() for name in enum_values.keys()}
    
    for i in range(num_cases):
        inputs = {}
        
        for name, type_ in args:
            # For enum-like string parameters, ensure all values are used at least once
            if name in enum_values and len(enum_values[name]) <= num_cases:
                valid_keys = enum_values[name]
                # Use each value at least once, then random
                if i < len(valid_keys):
                    value = valid_keys[i]
                    used_enum_values[name].add(value)
                else:
                    # After using all values, randomly sample
                    value = np.random.choice(valid_keys)
            # For spacetime tuples, generate varied separations (spacelike, timelike, lightlike)
            # Note: This is a simplified approach - for functions with pairs of tuples,
            # we generate tuples that when paired will produce different interval types
            elif name in spacetime_tuple_args and type_ == 'tuple':
                tuple_size = extract_tuple_size_from_docstring(docstring, name) or 4
                # Generate tuples that produce different interval types when paired
                # We'll generate base tuples and offsets to create varied separations
                base_tuple = tuple(np.random.uniform(-5.0, 5.0, size=tuple_size).tolist())
                
                # For different test cases, add offsets that create different interval types
                if i % 3 == 0:
                    # Spacelike: small time diff, large spatial diff
                    offset = [0.5, 8.0, 0.0, 0.0][:tuple_size]
                elif i % 3 == 1:
                    # Timelike: large time diff, small spatial diff
                    offset = [8.0, 0.5, 0.0, 0.0][:tuple_size]
                else:
                    # Lightlike: |t| ≈ |spatial|
                    t_val = np.random.uniform(3.0, 7.0)
                    spatial_mag = abs(t_val) * 0.95
                    offset = [t_val, spatial_mag, 0.0, 0.0][:tuple_size]
                
                value = tuple([base_tuple[j] + offset[j] for j in range(tuple_size)])
            # For arguments that check for zero, include some zero values
            elif name in args_needing_zero:
                arg_type_needing_zero = args_needing_zero[name]
                # Include zero in some test cases (about 20-30% of cases, at least 1 case)
                if (i < 2 and np.random.random() < 0.3) or (i == 0):
                    if arg_type_needing_zero == 'complex':
                        value = complex(0, 0)
                    elif arg_type_needing_zero == 'float':
                        value = 0.0
                    elif arg_type_needing_zero == 'int':
                        value = 0
                    else:
                        value = generate_test_value(type_, arg_name=name, code_str=code_str, docstring=docstring)
                else:
                    value = generate_test_value(type_, arg_name=name, code_str=code_str, docstring=docstring)
            else:
                value = generate_test_value(type_, arg_name=name, code_str=code_str, docstring=docstring)
            
            inputs[name] = value
        
        # Convert complex numbers and other non-JSON types to JSON-serializable format
        inputs = make_json_serializable(inputs)
        test_cases["test_cases"].append({
            "case_id": i + 1,
            "inputs": inputs
        })
    
    return test_cases

def create_function(code_str: str) -> Optional[callable]:
    """Create function from code string"""
    namespace = {
        'np': np,
        'Symbol': Symbol,
        'Function': Function,
        'FunctionClass': FunctionClass,
        'ProductReducedFunction': ProductReducedFunction
    }
    
    try:
        exec(extract_python_code(code_str), namespace)
        func_name = parse_function_signature(code_str)[0]
        return namespace[func_name]
    except Exception as e:
        # print(f"Error creating function: {e}")
        return None

def convert_from_json_serializable(value: Any, arg_type: str) -> Any:
    """
    Convert JSON-serializable values back to their original types.
    
    Args:
        value: JSON-serializable value (e.g., string representation of complex)
        arg_type: Type of the argument
        
    Returns:
        Converted value in original type
    """
    if arg_type == "complex":
        # Convert string like "(1+2j)" back to complex number
        if isinstance(value, str):
            try:
                # Remove parentheses if present
                value = value.strip('()')
                return complex(value)
            except (ValueError, TypeError):
                return value
        return value
    elif arg_type == "Symbol":
        return Symbol(value)
    elif arg_type == "FunctionClass":
        return ProductReducedFunction
    elif arg_type == "np.ndarray":
        return np.array(value)
    else:
        return value

def run_test_cases(test_cases: Dict, function: callable) -> Tuple[bool, Optional[str]]:
    """Run test cases for a function and return (success, error_message)"""
    try:
        for case in test_cases["test_cases"]:
            inputs = case["inputs"].copy()
            
            # Convert inputs to appropriate types
            for arg in test_cases["arguments"]:
                arg_name = arg["name"]
                arg_type = arg["type"]
                if arg_name in inputs:
                    inputs[arg_name] = convert_from_json_serializable(inputs[arg_name], arg_type)
            
            result = function(**inputs)
            # Store the result in the test case as string representation
            case["output"] = str(result)  # Convert sympy expression to string
            
            # Optionally, you can also store the type for reference
            case["output_type"] = type(result).__name__
            
            # print(f"\nTest case {case['case_id']}:")
            # print(f"Inputs: {inputs}")
            # print(f"Result: {result}")
        return True, None
    except Exception as e:
        error_msg = f"Error running test cases: {str(e)}"
        # print(error_msg)
        return False, error_msg
def generate_test_cases_for_problem(json_path: str) -> bool:
    """
    Wrapper function to generate test cases for a single problem file.
    
    Args:
        json_path: Path to the problem JSON file
        
    Returns:
        True if test generation was successful, False otherwise
    """
    try:
        errors = process_problem(json_path, run_tests=True)
        return len(errors) == 0
    except Exception as e:
        # Log error but don't raise - allow problem generation to continue
        print(f"Warning: Failed to generate test cases for {json_path}: {e}")
        return False


def repair_test_cases_for_problem(
    json_path: Path,
    threshold: int = 80,
    required_models: List[str] = None,
    dry_run: bool = False
) -> Tuple[bool, str]:
    """
    Repair test cases for a single problem using QC feedback.
    
    Args:
        json_path: Path to the problem JSON file
        threshold: Minimum score threshold for metrics
        required_models: List of model IDs that must have gradings
        dry_run: If True, don't actually modify the file
        
    Returns:
        Tuple of (success: bool, message: str)
    """
    if required_models is None:
        required_models = DEFAULT_REQUIRED_MODELS
    
    try:
        # Load problem data
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Check if this is a test-case-only failure
        is_failure, details = check_test_case_only_failure(
            data, threshold=threshold, required_models=required_models
        )
        
        if not is_failure:
            return False, "Not a test-case-only failure"
        
        # Extract review comments
        reviews = extract_test_case_reviews(data, required_models=required_models)
        
        if not reviews:
            return False, "No review comments found"
        
        if dry_run:
            return True, f"Would repair (has {len(reviews)} reviews)"
        
        # Extract problem details
        problem_details = data.get("problem_details", {})
        code_str = problem_details.get("Code", "")
        solution = problem_details.get("Solution", "")
        problem_statement = problem_details.get("Problem Statement", "")
        existing_test_cases = problem_details.get("test", {})
        
        # Parse function signature
        try:
            func_name, args, _ = parse_function_signature(code_str)
        except ValueError as e:
            return False, f"Failed to parse function signature: {e}"
        
        # Extract docstring
        docstring = extract_docstring_from_code(code_str)
        
        # Build repair prompt
        prompt = build_repair_prompt(
            problem_statement=problem_statement,
            solution=solution,
            code_str=code_str,
            func_name=func_name,
            args=args,
            existing_test_cases=existing_test_cases,
            review_comments=reviews,
            docstring=docstring
        )
        
        # Call AI to generate repaired test cases
        if not GENAI_AVAILABLE:
            return False, "GenAI module not available"

        # Add timeout for AI API call
        def ai_timeout_handler(signum, frame):
            raise TimeoutError(f"AI API call exceeded {REPAIR_TIMEOUT}s")

        try:
            api_key = read_api_key()

            # Set timeout for AI call
            old_handler = signal.signal(signal.SIGALRM, ai_timeout_handler)
            signal.alarm(REPAIR_TIMEOUT)

            try:
                response = call_gen_ai(prompt, api_key)
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)

        except TimeoutError as e:
            return False, f"AI call timed out: {e}"
        except Exception as e:
            return False, f"AI call failed: {e}"
        
        # Parse the response
        new_test_cases = parse_gemini_test_cases(response, func_name, args)
        
        if not new_test_cases or not new_test_cases.get("test_cases"):
            return False, "Failed to parse AI response"
        
        # Create the function and run tests
        function = create_function(code_str)
        if not function:
            return False, "Failed to create function from code"
        
        # Run test cases with timeout using signal
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Test execution exceeded {TIME_LIMIT}s")

        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(TIME_LIMIT)

        try:
            tests_passed, error_message = run_test_cases(new_test_cases, function)
        except TimeoutError as e:
            signal.alarm(0)  # Cancel the alarm
            signal.signal(signal.SIGALRM, old_handler)
            return False, str(e)
        except Exception as e:
            signal.alarm(0)  # Cancel the alarm
            signal.signal(signal.SIGALRM, old_handler)
            return False, f"Error running test cases: {str(e)}"
        finally:
            signal.alarm(0)  # Cancel the alarm
            signal.signal(signal.SIGALRM, old_handler)

        # Only save if tests passed - don't overwrite with broken test cases
        if not tests_passed:
            return False, f"New test cases failed to run: {error_message}"
        
        # Update the test cases in the data
        new_test_cases["tests_passed"] = tests_passed
        new_test_cases.pop("test_error", None)  # Tests passed, no error
        
        # Mark as repaired from QC feedback
        new_test_cases["repaired_from_qc"] = True
        new_test_cases["repair_timestamp"] = json.dumps({"timestamp": str(Path(json_path).stat().st_mtime)})
        
        # Preserve the old test cases before replacing
        if existing_test_cases and existing_test_cases.get("test_cases"):
            # Store old test cases for reference
            new_test_cases["previous_test_cases"] = existing_test_cases.get("test_cases", [])
            # Also keep track of previous repair history if it exists
            if existing_test_cases.get("previous_test_cases"):
                new_test_cases["previous_test_cases_history"] = existing_test_cases.get("previous_test_cases_history", [])
                new_test_cases["previous_test_cases_history"].append(existing_test_cases.get("previous_test_cases"))
        
        # Save back to file
        data["problem_details"]["test"] = new_test_cases
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        
        return True, f"Repaired successfully ({len(new_test_cases['test_cases'])} test cases)"
            
    except Exception as e:
        return False, f"Error repairing {json_path}: {e}"


def process_problem(json_path: str, run_tests: bool = True, force: bool = False, use_ai_generation: bool = True) -> List[str]:
    """Process a single problem file with optional LLM-assisted repair on failure or timeout"""
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        

        # Skip if tests already passed (unless force is True)
        if "test" in data["problem_details"] and not force:
            return []
        
        # If force is True, remove existing test entry to regenerate
        if force and "test" in data["problem_details"]:
            del data["problem_details"]["test"]

        code_str = data["problem_details"]["Code"]
        sol_str = data["problem_details"]["Solution"] if "Solution" in data["problem_details"] else ""
        problem_statement = data["problem_details"].get("Problem Statement", "")
        func_name, args, _ = parse_function_signature(code_str)
        # Extract docstring from code
        docstring = extract_docstring_from_code(code_str)
        # print(f'Processing {json_path} for function: {func_name} with args: {args}')
        
        # Try AI generation if enabled, fall back to rule-based if it fails
        test_cases = None
        if use_ai_generation:
            try:
                test_cases = generate_test_cases_with_gemini(
                    problem_statement=problem_statement,
                    solution=sol_str,
                    code_str=code_str,
                    func_name=func_name,
                    args=args,
                    docstring=docstring
                )
            except Exception as e:
                # Fall back to rule-based generation
                test_cases = None
        
        # Use rule-based generation if AI generation failed or wasn't requested
        if test_cases is None:
            test_cases = create_test_cases(func_name, args, code_str=code_str, docstring=docstring)
        
        # print(f"Generated test cases for {func_name}: {test_cases}")

        tests_passed = False
        error_message = None
        failed_once = False

        if run_tests:
            # print(f"Running tests for {json_path}...")
            max_attempts = 3
            for attempt in range(1, max_attempts + 1):
                try:
                    # Create the function from the code string
                    # print(f"\nAttempt {attempt} to run code in {json_path}...")
                    function = create_function(code_str)
                except Exception as e:
                    function = None
                    error_message = f"Error creating function: {str(e)}"
                    # print(f"Error creating function: {error_message}")
                if not function:
                    error_message = "Failed to create function"
                    
                if function is not None:
                    # print(f"Running test cases for {json_path} with timeout={TIME_LIMIT}s...")
                    from concurrent.futures import ProcessPoolExecutor, TimeoutError as FutureTimeout

                    try:
                        tests_passed, error_message = run_test_cases(test_cases, function)
                    except Exception as e:
                        tests_passed = False
                        error_message = str(e)
                    signal.alarm(0)

                    if tests_passed:
                        # print(f"Tests passed in attempt {attempt}")
                        break

                # on failure or timeout, attempt an LLM fix
                failed_once = True
                prompt = f"""Attempt {attempt}: fix the function based on error: {error_message}

                Solution for reference:
                {sol_str}

                Current code:
                {code_str}

                The code string will be converted to a python function through
                
                def extract_python_code(code_str: str) -> str:
                    #Extracts Python code from between \\begin{{python}} and \\end{{python}} tags
                    pattern = r"\\begin\\{{python\\}}(.*?)\\end\\{{python\\}}"
                    match = re.search(pattern, code_str, re.DOTALL)
                    return match.group(1).strip() if match else code_str

                                
                def create_function(code_str: str) -> Optional[callable]:
                    #Create function from code string
                    namespace = {{
                        'np': np,
                        'Symbol': Symbol,
                        'Function': Function,
                        'FunctionClass': FunctionClass,
                        'ProductRed[[ucedFunction': ProductReducedFunction
                    }}
                    
                    try:
                        exec(extract_python_code(code_str), namespace)
                        func_name = parse_function_signature(code_str)[0]
                        return namespace[func_name]
                    except Exception as e:
                        print(f"Error creating function: {{e}}")
                        return None
                
                Test cases are generated from the function signature using this function:

                def create_test_cases(func_name: str, args: List[Tuple[str, str]], 
                                    num_cases: int = 5) -> Dict:
                    #Generate test cases for a function
                    test_cases = {{
                        "function_name": func_name,
                        "arguments": [[{{"name": name, "type": type_}} for name, type_ in args]],
                        "test_cases": [[]
                    }}
                    
                    for i in range(num_cases):
                        inputs = {{name: generate_test_value(type_) for name, type_ in args}}
                        test_cases["test_cases"].append({{
                            "case_id": i + 1,
                            "inputs": inputs
                        }})
                    
                    return test_cases

                If there's an error resulting from the test cases, you should fix function signature so the code so that it compiles and runs correctly. 
                
                The function signature should never have a dictionary as an input, but rather a set of arguments that match the function signature with specified types.

                Your task is to fix the code so that it compiles and runs correctly.
                Please return the entire code block including imports, function definition, and any necessary changes.
                This code block serves as an answer to a problem, so you must NOT modify the answer, only fix the code so it compiles.

                ONLY return the code block, do not add any additional text."""
                # print(f"\nAttempt {attempt} to fix code with AI...")
                new_code_str = call_gen_ai(prompt, read_api_key())
                # print(f"New code from AI:\n{new_code_str}")
                new_code_str = extract_from_llm(new_code_str)
                new_code_str = '\\begin{python}\n' + new_code_str + '\n\\end{python}'
                # print(f"Extracted code:\n{new_code_str}")
                # if not new_code_str or new_code_str.strip() == code_str.strip():
                #     break
                code_str = new_code_str

        # record results
        # print(f"Final results for {json_path}:")
        # print(f"Tests passed: {tests_passed}")
        test_cases["tests_passed"] = tests_passed
        if tests_passed and failed_once:
            data["problem_details"]["Code"] = code_str
        if error_message:
            test_cases["test_error"] = error_message
        else:
            test_cases.pop("test_error", None)

        data["problem_details"]["test"] = test_cases
        with open(json_path, 'w', encoding="utf-8") as f:
            json.dump(data, f, indent=4)
            # print(f"Updated {json_path} with test results")

        if not tests_passed:
            # print(f"Tests failed for {json_path}: {error_message}")
            return [error_message or "Tests failed without a specific error"]
        return []

    except Exception as e:
        return [f"Error processing {json_path}: {e}"]

def run_existing_test_cases(json_path: str) -> List[str]:
    """Run existing test cases for a problem without generating new ones"""
    
    try:
        # Load the original file
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        # Check if test cases exist
        if "test" not in data["problem_details"]:
            return [f"No existing test cases found for {json_path}"]
            
        test_cases = data["problem_details"]["test"]
        code_str = data["problem_details"]["Code"]
        sol_str = data["problem_details"]["Solution"]
        function = create_function(code_str)
        
        if not function:
            return ["Failed to create function"]
            
        # Run the tests with up to 3 AI-assisted repair attempts on failure
        max_attempts = 3
        tests_passed = False
        error_message = None
        failed_once = False
        for attempt in range(1, max_attempts + 1):
            tests_passed, error_message = run_test_cases(test_cases, function)
            if tests_passed:
                break
            failed_once = True
            # Attempt to fix the code via AI
            prompt = f"""Attempt {attempt}: fix the function based on error: {error_message}

            Solution for reference: {sol_str}

            Current code: {code_str}

            Your task is to fix the code so that it compiles and runs correctly.
            Please return the entire code block including imports, function definition, and any necessary changes.
            This code block serves as an answer to a problem, so you must NOT modify the answer, only fix the code so it compiles.

            ONLY return the code block, do not add any additional text.
            """
            new_code_str = call_gen_ai(code_str, read_api_key())
            if not new_code_str or new_code_str == code_str:
                break
            code_str = new_code_str
            function = create_function(code_str)
        
        # Update test cases with final results
        test_cases["tests_passed"] = tests_passed
        if tests_passed and failed_once:
            data['problem_details']['Code'] = code_str  # Update code only if tests passed after fixing
        if not tests_passed and error_message:
            test_cases["test_error"] = error_message
        else:
            test_cases.pop("test_error", None)  # Remove error message if tests passed
            
        # Save updated test results back to original file
        data["problem_details"]["test"] = test_cases
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
            
        return [error_message] if error_message else []
        
    except Exception as e:
        return [f"Error processing {json_path}: {str(e)}"]

def process_file(json_path: Path, run_tests: bool = True, use_existing_tests: bool = False, update_outputs: bool = False, force: bool = False, use_ai_generation: bool = False) -> Tuple[str, List[str], Optional[Dict[str, str]]]:
    # print(f"\nProcessing {json_path}...")
    if update_outputs:
        errors = update_test_outputs(str(json_path))
    elif use_existing_tests:
        errors = run_existing_test_cases(str(json_path))
    else:
        errors = process_problem(str(json_path), run_tests, force=force, use_ai_generation=use_ai_generation)
    
    failed_info = None
    
    # Update log with test case attempt
    test_cases_log_path = ensure_test_log_file_exists(BASE_DIR)
    with open(test_cases_log_path, "r", encoding="utf-8") as f:
        try:
            test_cases_attempted = json.load(f)
            if not isinstance(test_cases_attempted, list):
                test_cases_attempted = []
        except Exception:
            test_cases_attempted = []

        # Append just this one path if it's not already recorded
        path_str = str(json_path)
        if path_str not in test_cases_attempted:
            test_cases_attempted.append(path_str)
            # Need to rewrite JSON to keep it valid (cannot stream-append to a JSON array)
            with open(test_cases_log_path, "w", encoding="utf-8") as wf:
                json.dump(test_cases_attempted, wf, indent=2)
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if not data["problem_details"].get("test", {}).get("tests_passed", True):
                failed_info = {
                    'path': str(json_path),
                    'error': data["problem_details"].get("test", {}).get("test_error", "Unknown error")
                }
    except Exception:
        pass

    return str(json_path), errors, failed_info

def process_directory(
    directory: str,
    run_tests: bool = True,
    use_existing_tests: bool = False,
    update_outputs: bool = False,
    num_workers: int = MAX_WORKERS,
    force: bool = False,
    use_ai_generation: bool = False,
) -> Dict[str, List[str]]:
    json_files = list(Path(directory).glob("*.json"))
    test_cases_log_path = ensure_test_log_file_exists(BASE_DIR)
    with open(test_cases_log_path, "r", encoding="utf-8") as f:
        try:
            test_cases_attempted = json.load(f)
            if not isinstance(test_cases_attempted, list):
                test_cases_attempted = []
        except Exception:
            test_cases_attempted = []
    json_to_process = []
    
    
    for p in json_files:
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            pd = data.get("problem_details", {})
            # Include file if it has no tests, or if force is True
            if "test" not in pd or force:
                json_to_process.append(p)            
        except Exception:
            # Skip files that can't be read
            continue

    json_files = json_to_process
    print(f"Found {len(json_files)} JSON files requiring test cases in {directory}")

    errors = {}
    with ProcessPoolExecutor(max_workers=num_workers) as exe, \
            tqdm(total=len(json_files), desc="Generating tests", unit="file") as pbar:
        future_to_path = {
            exe.submit(process_file, p, run_tests, use_existing_tests, update_outputs, force, use_ai_generation): p
            for p in json_files
        }
        for fut in as_completed(future_to_path):
            path = future_to_path[fut]
            try:
                file_path, file_errors, failed_info = fut.result(timeout=TIME_LIMIT)
            except TimeoutError:
                file_errors = [f"Processing exceeded {TIME_LIMIT}s"]
                fut.cancel()
            except Exception as e:
                file_errors = [str(e)]
            if file_errors:
                errors[str(path)] = file_errors
            pbar.update(1)
    return errors


def process_repair_directory(
    directory: Path,
    threshold: int = 80,
    required_models: List[str] = None,
    num_workers: int = 1,  # Sequential by default for API rate limits
    dry_run: bool = False,
    output_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Process directory to repair test-case-only failures using QC feedback.
    
    Args:
        directory: Directory containing problem JSON files
        threshold: Minimum score threshold for metrics
        required_models: List of model IDs that must have gradings
        num_workers: Number of parallel workers (default 1 for API rate limiting)
        dry_run: If True, just report what would be repaired
        output_dir: If provided, copy successfully repaired files to this directory
        
    Returns:
        Dictionary with repair results
    """
    if required_models is None:
        required_models = DEFAULT_REQUIRED_MODELS
    
    directory = Path(directory)
    
    # Create output directory if specified
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # First, identify all test-case-only failures
    print(f"\nScanning {directory} for test-case-only failures...")
    failures = identify_test_case_only_failures(
        directory,
        threshold=threshold,
        required_models=required_models
    )

    print(f"Found {len(failures)} problems with test-case-only failures")

    # Filter out files that already exist in output_dir
    if output_dir:
        original_count = len(failures)
        failures = [
            (json_path, details)
            for json_path, details in failures
            if not (output_dir / json_path.name).exists()
        ]
        skipped_count = original_count - len(failures)
        if skipped_count > 0:
            print(f"Skipping {skipped_count} files that already exist in output directory")
            print(f"Remaining to process: {len(failures)} files")
    
    if not failures:
        return {
            "total_found": 0,
            "repaired": [],
            "failed": [],
            "skipped": [],
            "copied": []
        }
    
    results = {
        "total_found": len(failures),
        "repaired": [],
        "failed": [],
        "skipped": [],
        "copied": []
    }
    
    if dry_run:
        print("\n[DRY RUN] Would repair the following files:")
        for json_path, details in failures:
            print(f"  - {json_path.name}")
            for tc_failure in details.get("test_case_failures", []):
                print(f"    • {tc_failure['model_id']}: score={tc_failure['value']}")
                if tc_failure.get('comment'):
                    # Truncate long comments
                    comment = tc_failure['comment'][:100] + "..." if len(tc_failure['comment']) > 100 else tc_failure['comment']
                    print(f"      \"{comment}\"")
        if output_dir:
            print(f"\n[DRY RUN] Successfully repaired files would be copied to: {output_dir}")
        return results
    
    # Process each failure
    print("\nRepairing test cases...")

    if num_workers == 1:
        # Sequential processing
        for json_path, details in tqdm(failures, desc="Repairing", unit="file"):
            success, message = repair_test_cases_for_problem(
                json_path,
                threshold=threshold,
                required_models=required_models,
                dry_run=False
            )

            if success:
                results["repaired"].append({
                    "path": str(json_path),
                    "message": message
                })
                print(f"  ✓ {json_path.name}: {message}")

                # Copy to output directory if specified
                if output_dir:
                    dest_path = output_dir / json_path.name
                    shutil.copy2(json_path, dest_path)
                    results["copied"].append(str(dest_path))
                    print(f"    → Copied to {dest_path}")
            else:
                results["failed"].append({
                    "path": str(json_path),
                    "message": message
                })
                print(f"  ✗ {json_path.name}: {message}")
    else:
        # Parallel processing
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Submit all repair tasks
            future_to_path = {
                executor.submit(
                    repair_test_cases_for_problem,
                    json_path,
                    threshold,
                    required_models,
                    False
                ): (json_path, details)
                for json_path, details in failures
            }

            # Process completed repairs with progress bar
            for future in tqdm(as_completed(future_to_path, timeout=REPAIR_TIMEOUT), total=len(failures), desc="Repairing", unit="file"):
                json_path, details = future_to_path[future]
                try:
                    success, message = future.result(timeout=REPAIR_TIMEOUT)

                    if success:
                        results["repaired"].append({
                            "path": str(json_path),
                            "message": message
                        })
                        print(f"  ✓ {json_path.name}: {message}")

                        # Copy to output directory if specified
                        if output_dir:
                            dest_path = output_dir / json_path.name
                            shutil.copy2(json_path, dest_path)
                            results["copied"].append(str(dest_path))
                            print(f"    → Copied to {dest_path}")
                    else:
                        results["failed"].append({
                            "path": str(json_path),
                            "message": message
                        })
                        print(f"  ✗ {json_path.name}: {message}")
                except TimeoutError:
                    results["failed"].append({
                        "path": str(json_path),
                        "message": f"Repair timed out after {REPAIR_TIMEOUT} seconds"
                    })
                    print(f"  ✗ {json_path.name}: Timed out after {REPAIR_TIMEOUT}s")
                    future.cancel()
                except Exception as e:
                    results["failed"].append({
                        "path": str(json_path),
                        "message": f"Exception during repair: {str(e)}"
                    })
                    print(f"  ✗ {json_path.name}: Exception - {str(e)}")
    
    # Summary
    print(f"\n{'='*60}")
    print("Repair Summary:")
    print(f"  Total found: {results['total_found']}")
    print(f"  Repaired: {len(results['repaired'])}")
    print(f"  Failed: {len(results['failed'])}")
    if output_dir:
        print(f"  Copied to {output_dir}: {len(results['copied'])}")
    
    return results


def collect_output_types(directory: str) -> Tuple[Dict[str, int], Dict[str, List[str]]]:
    """
    Collect all unique output types from test cases in the directory
    Returns:
        - Dict mapping output_types to their counts
        - Dict mapping output_types to lists of problem_ids (for types with <20 occurrences)
    """
    output_types = {}
    rare_types_problems = {}  # Maps output_type to list of problem_ids
    
    for json_path in Path(directory).glob("*.json"):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            test_data = data.get("problem_details", {}).get("test", {})
            if not test_data:
                continue
            
            problem_id = data.get("problem_id", str(json_path))
                
            for case in test_data.get("test_cases", []):
                output_type = case.get("output_type")
                if output_type:
                    output_types[output_type] = output_types.get(output_type, 0) + 1
                    # Track problem_ids for types that might end up being rare
                    if output_type not in rare_types_problems:
                        rare_types_problems[output_type] = set()
                    rare_types_problems[output_type].add(problem_id)
                    
        except Exception as e:
            print(f"Error processing {json_path}: {str(e)}")
    
    # Filter out types that appear ≥20 times
    rare_types_problems = {
        type_name: sorted(list(problems))
        for type_name, problems in rare_types_problems.items()
        if output_types[type_name] < 20
    }
            
    return output_types, rare_types_problems


# ============================================================================
# QC-Feedback Repair Functions
# ============================================================================

REQUIRED_METRICS = [
    "problem_quality",
    "solution_completeness",
    "solution_quality",
    "test_case_quality"
]

DEFAULT_REQUIRED_MODELS = ["pipeline-review"]


def get_latest_grading(data: Dict[str, Any], model_id: str) -> Optional[Dict[str, Any]]:
    """Get the latest (highest index) grading for a model."""
    gradings = data.get("quality_gradings", {}).get(model_id, [])
    if not gradings:
        return None
    
    # Find grading with highest index
    latest = max(gradings, key=lambda g: g.get("index", -1))
    return latest


def check_test_case_only_failure(
    data: Dict[str, Any],
    threshold: int = 80,
    required_models: List[str] = None
) -> Tuple[bool, Dict[str, Any]]:
    """
    Check if a problem fails ONLY on test_case_quality (all other metrics pass).
    
    Args:
        data: Problem JSON data
        threshold: Minimum score for each metric
        required_models: List of model IDs that must have gradings
        
    Returns:
        Tuple of (is_test_case_only_failure, details)
    """
    if required_models is None:
        required_models = DEFAULT_REQUIRED_MODELS
    
    details = {
        "has_all_models": False,
        "test_case_failures": [],
        "other_failures": [],
        "is_test_case_only_failure": False
    }
    
    # Check each required model
    models_present = []
    for model_id in required_models:
        latest = get_latest_grading(data, model_id)
        if latest is None:
            continue
        
        models_present.append(model_id)
        
        # Check each metric
        for metric in REQUIRED_METRICS:
            value = latest.get(metric)
            if value is None:
                continue
            
            if value < threshold:
                if metric == "test_case_quality":
                    details["test_case_failures"].append({
                        "model_id": model_id,
                        "value": value,
                        "comment": latest.get("test_case_quality_comment", "")
                    })
                else:
                    details["other_failures"].append({
                        "model_id": model_id,
                        "metric": metric,
                        "value": value
                    })
    
    details["has_all_models"] = len(models_present) == len(required_models)
    
    # It's a test-case-only failure if:
    # 1. All required models are present
    # 2. There are test_case_quality failures
    # 3. There are NO other metric failures
    details["is_test_case_only_failure"] = (
        details["has_all_models"] and
        len(details["test_case_failures"]) > 0 and
        len(details["other_failures"]) == 0
    )
    
    return details["is_test_case_only_failure"], details


def identify_test_case_only_failures(
    source_dir: Path,
    threshold: int = 80,
    required_models: List[str] = None
) -> List[Tuple[Path, Dict[str, Any]]]:
    """
    Find problems that fail ONLY on test_case_quality (all other metrics pass).
    
    Args:
        source_dir: Directory containing problem JSON files
        threshold: Minimum score for each metric (default: 80)
        required_models: List of model IDs that must have gradings
        
    Returns:
        List of (file_path, details) tuples for problems with test-case-only failures
    """
    if required_models is None:
        required_models = DEFAULT_REQUIRED_MODELS
    
    failures = []
    
    json_files = sorted([f for f in source_dir.glob("*.json") 
                        if f.name not in ("assignment.json", "config.json")])
    
    for json_path in json_files:
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            is_failure, details = check_test_case_only_failure(
                data, threshold=threshold, required_models=required_models
            )
            
            if is_failure:
                failures.append((json_path, details))
                
        except Exception as e:
            print(f"Error reading {json_path}: {e}")
            continue
    
    return failures


def extract_test_case_reviews(
    data: Dict[str, Any],
    required_models: List[str] = None
) -> List[Dict[str, Any]]:
    """
    Extract all test_case_quality reviews from quality_gradings.
    
    Args:
        data: Problem JSON data
        required_models: List of model IDs to extract reviews from
        
    Returns:
        List of review dictionaries with model_id, index, score, and comment
    """
    if required_models is None:
        required_models = DEFAULT_REQUIRED_MODELS
    
    reviews = []
    quality_gradings = data.get("quality_gradings", {})
    
    for model_id in required_models:
        model_gradings = quality_gradings.get(model_id, [])
        
        for grading in model_gradings:
            review = {
                "model_id": model_id,
                "index": grading.get("index", 0),
                "timestamp": grading.get("timestamp", ""),
                "test_case_quality": grading.get("test_case_quality"),
                "test_case_quality_comment": grading.get("test_case_quality_comment", "")
            }
            reviews.append(review)
    
    return reviews


def update_test_outputs(json_path: str) -> List[str]:
    """Update outputs for existing test cases by running the code with existing inputs"""
    try:
        # Load the original file
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        # Check if test cases exist
        if "test" not in data["problem_details"]:
            return [f"No existing test cases found for {json_path}"]
            
        test_cases = data["problem_details"]["test"]
        code_str = data["problem_details"]["Code"]
        function = create_function(code_str)
        
        if not function:
            return ["Failed to create function"]
            
        # Run the tests with existing inputs and update outputs
        try:
            for case in test_cases["test_cases"]:
                inputs = case["inputs"].copy()
                
                # Convert inputs to appropriate types
                for arg in test_cases["arguments"]:
                    arg_name = arg["name"]
                    arg_type = arg["type"]
                    if arg_name in inputs:
                        inputs[arg_name] = convert_from_json_serializable(inputs[arg_name], arg_type)
                
                result = function(**inputs)
                # Update the output in the test case
                case["output"] = str(result)
                case["output_type"] = type(result).__name__
                
                # print(f"\nTest case {case['case_id']}:")
                # print(f"Inputs: {inputs}")
                # print(f"Updated Result: {result}")
            
            # Mark tests as passed and remove any error message
            test_cases["tests_passed"] = True
            test_cases.pop("test_error", None)
            
            # Save updated test results back to original file
            data["problem_details"]["test"] = test_cases
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4)
                
            return []
            
        except Exception as e:
            error_msg = f"Error running test cases: {str(e)}"
            test_cases["tests_passed"] = False
            test_cases["test_error"] = error_msg
            
            # Save the error state
            data["problem_details"]["test"] = test_cases
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4)
                
            return [error_msg]
            
    except Exception as e:
        return [f"Error processing {json_path}: {str(e)}"]



def main(
    base_dir: Path = Path('.'),
    directory: Optional[Path] = None,
    num_workers: int = MAX_WORKERS,
    use_existing_tests: bool = False,
    collect_types: bool = False,
    update_outputs: bool = False,
    force: bool = False,
    use_ai_generation: bool = False,
    repair_from_qc: bool = False,
    qc_threshold: int = 80,
    dry_run: bool = False,
    output_dir: Optional[Path] = None,
):
    """
    Main function for generating test cases.
    
    Args:
        base_dir: Base directory (used if directory is not provided, looks for json_finished_problems subdirectory)
        directory: Direct path to directory containing problem JSON files (for retroactive processing)
        num_workers: Number of parallel workers
        use_existing_tests: Whether to use existing test cases
        collect_types: Whether to collect output types
        update_outputs: Whether to update test outputs
        force: Whether to force regeneration
        use_ai_generation: Whether to use AI for test case generation
        repair_from_qc: Whether to repair test cases using QC feedback
        qc_threshold: Threshold for QC metrics when repairing
        dry_run: Whether to run in dry-run mode
    """
    if directory is None:
        directory = Path(base_dir) / "json_finished_problems"
    else:
        directory = Path(directory)
    
    global BASE_DIR
    BASE_DIR = Path(base_dir).resolve()
    try:
        # Handle repair mode first
        if repair_from_qc:
            print("=" * 60)
            print("QC-Feedback Test Case Repair Mode")
            print("=" * 60)
            print(f"Directory: {directory}")
            print(f"Threshold: {qc_threshold}")
            print(f"Dry run: {dry_run}")
            print("=" * 60)
            
            results = process_repair_directory(
                directory=directory,
                threshold=qc_threshold,
                num_workers=num_workers,
                dry_run=dry_run,
                output_dir=output_dir
            )
            
            if dry_run:
                print(f"\n[DRY RUN] Found {results['total_found']} problems that would be repaired")
            else:
                print(f"\nRepair complete: {len(results['repaired'])} repaired, {len(results['failed'])} failed")
            return
        
        if collect_types:
            output_types, rare_types_problems = collect_output_types(directory)
            print("\n=== Output Types Summary ===")
            print(f"Found {len(output_types)} different output types:")
            for type_name, count in sorted(output_types.items()):
                print(f"- {type_name}: {count} occurrences")
                if type_name in rare_types_problems:
                    print("  Problems with this type:")
                    for problem_id in rare_types_problems[type_name]:
                        print(f"    * {problem_id}")
        else:
            if update_outputs:
                errors = process_directory(
                    directory, use_existing_tests=True, update_outputs=True, num_workers=num_workers, force=force, use_ai_generation=use_ai_generation
                )
            else:
                errors = process_directory(
                    directory, use_existing_tests=use_existing_tests, num_workers=num_workers, force=force, use_ai_generation=use_ai_generation
                )

            if not errors:
                print("\nAll problems processed successfully!")
            else:
                print(f"\nEncountered errors in {len(errors)} problems.")
                print("See error_report.json for details.")
    except Exception as e:
        print(f"Fatal error: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate test cases for problem JSON files"
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default=".",
        help="Base directory (default: current directory). Used if --directory is not provided."
    )
    parser.add_argument(
        "--directory",
        type=str,
        default=None,
        help="Direct path to directory containing problem JSON files (for retroactive processing). "
             "If not provided, looks for json_finished_problems subdirectory in base-dir."
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=MAX_WORKERS,
        help=f"Number of parallel workers (default: {MAX_WORKERS})"
    )
    parser.add_argument(
        "--use-existing-tests",
        action="store_true",
        help="Use existing test cases instead of generating new ones"
    )
    parser.add_argument(
        "--collect-types",
        action="store_true",
        help="Collect and display output types summary"
    )
    parser.add_argument(
        "--update-outputs",
        action="store_true",
        help="Update outputs for existing test cases"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force regeneration of test cases, overwriting existing ones"
    )
    parser.add_argument(
        "--use-ai-generation",
        action="store_true",
        help="Use Gemini API to generate test cases (falls back to rule-based if AI generation fails)"
    )
    parser.add_argument(
        "--repair-from-qc",
        action="store_true",
        help="Repair test cases using QC review feedback (for test-case-only failures)"
    )
    parser.add_argument(
        "--qc-threshold",
        type=int,
        default=80,
        help="Threshold for QC metrics when using --repair-from-qc (default: 80)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show which files would be processed without making changes"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Copy successfully repaired files to this directory (use with --repair-from-qc)"
    )
    
    args = parser.parse_args()
    
    main(
        base_dir=Path(args.base_dir),
        directory=Path(args.directory) if args.directory else None,
        num_workers=args.num_workers,
        use_existing_tests=args.use_existing_tests,
        collect_types=args.collect_types,
        update_outputs=args.update_outputs,
        force=args.force,
        use_ai_generation=args.use_ai_generation,
        repair_from_qc=args.repair_from_qc,
        qc_threshold=args.qc_threshold,
        dry_run=args.dry_run,
        output_dir=Path(args.output_dir) if args.output_dir else None,
    )
