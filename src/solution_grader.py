from datetime import datetime, timezone
import signal
from contextlib import contextmanager
import threading

from pathlib import Path
from typing import Dict, List, Optional, Callable, Tuple, Any
from sympy import Symbol, Function, FunctionClass, sympify, Mul
from tqdm import tqdm
import simplejson as json  # for compatibility

import re
import numpy as np
import math
import ast
import multiprocessing
from queue import Empty

from .problem_processor_base import BaseProblemProcessor



########### Helper code for code verification grading ###########

class TimeoutException(Exception):
    pass


@contextmanager
def timeout(seconds):
    # signal.alarm only works in the main thread; no-op otherwise
    if threading.current_thread() is not threading.main_thread():
        try:
            yield
        finally:
            pass
    else:
        def timeout_handler(signum, frame):
            raise TimeoutException("Function call timed out")

        # Register the signal function handler
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(seconds)

        try:
            yield
        finally:
            # Disable the alarm
            signal.alarm(0)


class ProductReducedFunction(Function):
    @classmethod
    def eval(cls, *args):
        args = tuple(map(sympify, args))
        multiplied_args = Mul(*args)
        if args[0] != multiplied_args:
            return cls(multiplied_args)


def _execute_grading_task(attempt_code: str, test: Dict, standard_results: List[Dict],
                         timeout_seconds: int, result_queue: multiprocessing.Queue,
                         quiet: bool = False):
    """
    Executes the grading logic for a single attempt code in a separate process.
    Creates the attempt function, runs test cases, compares results, and puts
    the outcome into the result_queue.
    """
    grader = CodeVerificationGrader(quiet=quiet)  # Minimal instance for helper methods
    grader.timeout_seconds = timeout_seconds  # Ensure timeout is set

    try:
        if not quiet:
            print(f"DEBUG (Process {multiprocessing.current_process().pid}): Creating attempt function...")
        attempt_func = grader.create_function(attempt_code)
        if not quiet:
            print(
                f"DEBUG (Process {multiprocessing.current_process().pid}): Attempt function created: {bool(attempt_func)}")
        if not attempt_func:
            result_queue.put({
                "verified": 0,
                "error": "Could not create attempt function",
                "test_cases": []
            })
            return

        if not quiet:
            print(f"DEBUG (Process {multiprocessing.current_process().pid}): Running test cases for attempt function...")
        attempt_success, attempt_results = grader.run_test_cases(test, attempt_func)
        if not quiet:
            print(f"DEBUG (Process {multiprocessing.current_process().pid}): Attempt test cases success: {attempt_success}")
        if not attempt_success:
            # run_test_cases already includes timeout/error details in its print,
            # return a generic error here or enhance run_test_cases to return error details
            result_queue.put({
                "verified": 0,
                "error": "Error or timeout running attempt test cases",
                "test_cases": []  # Or potentially include partial results if needed
            })
            return

        # Compare results
        if not quiet:
            print(f"DEBUG (Process {multiprocessing.current_process().pid}): Comparing results...")
        all_match = True
        test_cases_details = []

        for std_case, att_case, test_case_config in zip(standard_results, attempt_results, test["test_cases"]):
            std_output = std_case["output"]
            att_output = att_case["output"]
            tolerance = test_case_config.get("tolerance")
            matches = bool(grader._compare_outputs(std_output, att_output, tolerance=tolerance))

            test_cases_details.append({
                "inputs": std_case["inputs"],
                "standard_output": str(std_output),
                "attempt_output": _format_output(att_output),
                "matches": matches,
                "tolerance_used": tolerance
            })

            if not matches:
                all_match = False

        if all_match:
            if not quiet:
                print(f"DEBUG (Process {multiprocessing.current_process().pid}): Found matching solution.")
            result_queue.put({
                "verified": 1,
                "test_cases": test_cases_details,
                "matched_code": attempt_code
            })
        else:
            if not quiet:
                print(f"DEBUG (Process {multiprocessing.current_process().pid}): No match found.")
            result_queue.put({
                "verified": 0,
                "error": "Outputs do not match standard solution",
                "test_cases": test_cases_details
            })

    except Exception as e:
        if not quiet:
            print(f"DEBUG (Process {multiprocessing.current_process().pid}): Exception in _execute_grading_task: {str(e)}")
        result_queue.put({
            "verified": 0,
            "error": f"Exception during grading task: {str(e)}",
            "test_cases": []
        })
    finally:
        if not quiet:
            print(f"DEBUG (Process {multiprocessing.current_process().pid}): Exiting _execute_grading_task.")


def _format_output(obj, tol=1e-6):
    """
    Recursively convert obj into a JSON-friendly structure of strings,
    formatting every float/int with precision matching tol.
    """
    # figure out how many decimal places tol implies
    # e.g. tol=1e-6 → places=6; tol=1e-2 → places=2
    try:
        places = max(0, int(round(-math.log10(tol))))
    except (ValueError, ZeroDivisionError):
        places = 6  # fallback if tol is 0 or weird

    # numbers → formatted string
    if isinstance(obj, (float, int)):
        return f"{obj:.{places}f}"

    # numpy scalars
    try:
        import numpy as np
        if isinstance(obj, (np.floating, np.integer)):
            return f"{float(obj):.{places}f}"
    except ImportError:
        pass

    # numpy arrays → convert to nested lists
    if isinstance(obj, np.ndarray):
        return [_format_output(x, tol) for x in obj.tolist()]

    # lists/tuples
    if isinstance(obj, (list, tuple)):
        return [_format_output(x, tol) for x in obj]

    # dicts
    if isinstance(obj, dict):
        return {k: _format_output(v, tol) for k, v in obj.items()}

    # fallback for anything else (sympy, custom objects, etc)
    return str(obj)


class CodeVerificationGrader(BaseProblemProcessor):
    def __init__(
            self,
            model_name: str = "code_verifier",  # Not used for LLM calls, only for BaseProblemProcessor compatibility
            config_path: str = "configs/grading_config.json",
            problems_dir: str = "data/tpbench",
            multi_gpu: int = 1,  # Not used, only for BaseProblemProcessor compatibility
            timeout_seconds: int = 10,
            quiet: bool = False,
    ):
        super().__init__(
            model_name=model_name,
            config_path=config_path,
            problems_dir=problems_dir,
            multi_gpu=multi_gpu,
            quiet=quiet,
        )
        self.timeout_seconds = timeout_seconds
        np.seterr(all='warn')
    
    def _init_client(self):
        """Override to skip model loading since CodeVerificationGrader doesn't need an LLM."""
        # Code verification doesn't use any LLM, so we skip the model initialization
        # Just initialize api_client to None to avoid errors
        self.api_client = None
        self.local_llm = None
        self.local_tokenizer = None

    def _grade_problem(self, problem_path, regrade_all: bool) -> None:
        """Grade a single problem using code verification."""
        problem = self._load_problem(problem_path)
        print(f"Processing problem {problem_path} with code verifier...")
        for model_solution in problem["model_solutions"]:
            for idx, attempt in enumerate(model_solution["attempts"]):
                # If regrade_all is True, clear existing verifier_result
                if regrade_all:
                    attempt["verifier_result"] = {}

                # Determine if the attempt needs grading
                # For code grading, check if dict is empty/missing
                verifier_result = attempt.get("verifier_result")
                # Handle backward compatibility: convert old list format to dict
                if isinstance(verifier_result, list) and verifier_result:
                    verifier_result = verifier_result[0]
                    attempt["verifier_result"] = verifier_result
                
                needs_grading = not verifier_result or not isinstance(verifier_result, dict)
                if not needs_grading:
                    print(f"DEBUG: Attempt {idx + 1} already graded. Skipping...")
                    continue

                if needs_grading:
                    print(
                        f"DEBUG: Grading problem '{problem_path}', model '{model_solution['model']}', attempt {idx + 1} (index {idx})...")
                    evaluation = self._grade_solution(attempt, problem, model_solution["model"])
                    print(f"DEBUG: Finished grading attempt {idx + 1}.")
                    evaluation["timestamp"] = datetime.now(timezone.utc).isoformat()
                    
                    # Ensure grading_model is not present for code grading
                    evaluation.pop("grading_model", None)

                    # Store as dict directly (not a list)
                    attempt["verifier_result"] = evaluation

        self._save_problem(problem, problem_path)

    def _extract_function_name(self, code: str) -> Optional[str]:
        """Extract the first function name from Python code"""
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    return node.name
            return None
        except Exception as e:
            print(f"Error parsing code: {str(e)}")
            return None

    def _extract_python_code(self, text: str) -> Optional[str]:
        """Extract Python code from text between either:
        1. \\begin{python} and \\end{python}
        2. ```python and ```

        Find all matches of both formats and return the last matching code block.
        Clean up any nested markers.
        """
        if not text:
            return None

        all_matches = []

        # Find all \\begin{python} format matches
        pattern1 = r"\\begin{python}(.*?)\\end{python}"
        matches1 = re.findall(pattern1, text, re.DOTALL)
        all_matches.extend((match.strip(), 1) for match in matches1)

        # Find all ```python format matches
        pattern2 = r"```python\s*(.*?)```"
        matches2 = re.findall(pattern2, text, re.DOTALL)
        # Clean up any nested markers for ```python format
        cleaned_matches2 = []
        for match in matches2:
            clean_code = re.sub(r"```python\s*|\s*```", "", match, flags=re.DOTALL)
            if clean_code.strip():
                cleaned_matches2.append((clean_code.strip(), 2))
        all_matches.extend(cleaned_matches2)

        # Return the last match if any matches found
        if all_matches:
            return all_matches[-1][0]

        return None

    def _extract_attempt_codes(self, text: str) -> List[str]:
        """Extract all Python code blocks from attempt solution."""
        if not text:
            return []

        code_blocks = []

        # Find all \\begin{python} format matches
        pattern1 = r"\\begin{python}(.*?)\\end{python}"
        matches1 = re.findall(pattern1, text, re.DOTALL)
        for match in matches1:
            if match.strip():
                code_blocks.append(match.strip())

        # Find all ```python format matches
        pattern2 = r"```python\s*(.*?)```"
        matches2 = re.findall(pattern2, text, re.DOTALL)
        for match in matches2:
            clean_code = re.sub(r"```python\s*|\s*```", "", match, flags=re.DOTALL)
            if clean_code.strip():
                code_blocks.append(clean_code.strip())

        # Return only the final code block as a list
        if code_blocks:
            return [code_blocks[-1]]
        return []

    def create_function(self, code_str: str) -> Optional[callable]:
        """Create function from code string"""
        namespace = {
            'np': np,
            'Symbol': Symbol,
            'Function': Function,
            'FunctionClass': FunctionClass,
            'ProductReducedFunction': ProductReducedFunction,
            'sqrt': np.sqrt,
            'log': np.log,
            'exp': np.exp,
            'math': __import__('math')
        }

        try:
            if not self.quiet:
                print("DEBUG: --- Attempt Code Start ---")
                print(code_str)
                print("DEBUG: --- Attempt Code End ---")
                print(f"DEBUG (PID {multiprocessing.current_process().pid}): Executing code string...")  # Modified print
            
            # Pre-processing to fix Python 3.8 type hint incompatibility
            # Replace 'tuple[' with 'Tuple[' and ensure Tuple is imported
            if 'tuple[' in code_str:
                if 'from typing import' not in code_str:
                    code_str = 'from typing import Tuple\n' + code_str
                elif 'Tuple' not in code_str:
                    code_str = code_str.replace('from typing import', 'from typing import Tuple,')

                # Replace tuple[...] with Tuple[...]
                # Use regex to only replace tuple when used as a type hint (subscriptable)
                code_str = re.sub(r'tuple\[', 'Tuple[', code_str)

                if not self.quiet:
                    print("DEBUG: Applied type hint fix (tuple -> Tuple)")
                    print(code_str)

            exec(code_str, namespace)
            if not self.quiet:
                print(f"DEBUG (PID {multiprocessing.current_process().pid}): Execution finished.")  # Modified print
            func_name = self._extract_function_name(code_str)
            if not self.quiet:
                print(
                    f"DEBUG (PID {multiprocessing.current_process().pid}): Extracted function name: {func_name}")  # Modified print
            if func_name and func_name in namespace:
                return namespace[func_name]
            else:
                if not self.quiet:
                    print(
                        f"DEBUG (PID {multiprocessing.current_process().pid}): Function '{func_name}' not found in namespace after exec.")  # Modified print
                return None
        except Exception as e:
            if not self.quiet:
                print(f"Error creating function (PID {multiprocessing.current_process().pid}): {e}")  # Modified print
            return None

    def run_test_cases(self, test: Dict, function: callable) -> Tuple[bool, List[Dict]]:
        # Keep the debug prints here, they run inside the subprocess
        if not self.quiet:
            print(f"DEBUG (PID {multiprocessing.current_process().pid}): Entering run_test_cases")  # Modified print
        try:
            results = []
            if not self.quiet:
                print(
                    f"DEBUG (PID {multiprocessing.current_process().pid}): Starting loop for {len(test['test_cases'])} test cases.")  # Modified print
            for i, case in enumerate(test["test_cases"]):
                inputs = case["inputs"].copy()
                if not self.quiet:
                    print(
                        f"DEBUG (PID {multiprocessing.current_process().pid}): Running test case {i + 1}/{len(test['test_cases'])} with inputs: {inputs}")  # Modified print
                # Convert inputs to appropriate types
                for arg in test["arguments"]:
                    if arg["type"] == "Symbol":
                        inputs[arg["name"]] = Symbol(inputs[arg["name"]])
                    elif arg["type"] == "FunctionClass":
                        inputs[arg["name"]] = ProductReducedFunction
                    elif arg["type"] == "np.ndarray":
                        inputs[arg["name"]] = np.array(inputs[arg["name"]])
                    elif arg["type"] == "complex":
                        # Convert string representation of complex number to complex type
                        if isinstance(inputs[arg["name"]], str):
                            try:
                                # Handle format like "(1+2j)" or "1+2j"
                                clean_val = inputs[arg["name"]].strip().replace('(', '').replace(')', '').replace(' ', '')
                                inputs[arg["name"]] = complex(clean_val)
                            except ValueError:
                                pass # Keep as is if conversion fails


                try:
                    # Keep the inner timeout as a first defense, but the process timeout is the main guard
                    with timeout(self.timeout_seconds):
                        # Keep np.errstate here as well
                        with np.errstate(all='warn'):
                            if not self.quiet:
                                print(
                                    f"DEBUG (PID {multiprocessing.current_process().pid}): Calling function for case {i + 1}...")  # Modified print
                            result = function(**inputs)
                            if not self.quiet:
                                print(
                                    f"DEBUG (PID {multiprocessing.current_process().pid}): Function call finished for case {i + 1}.")  # Modified print

                            # Check for nan/inf values
                            if isinstance(result, (float, np.ndarray)) and (
                                    np.isnan(result).any() or np.isinf(result).any()):
                                raise ValueError(f"Function produced nan/inf values for inputs: {inputs}")

                            case_result = case.copy()
                            case_result["output"] = result
                            results.append(case_result)
                except TimeoutException:
                    # This timeout might still happen if the function is pure Python and loops
                    error_msg = f"DEBUG (PID {multiprocessing.current_process().pid}): Function execution timed out (inner signal) after {self.timeout_seconds} seconds with inputs: {inputs}"  # Modified print
                    if not self.quiet:
                        print(error_msg)
                    return False, []  # Existing return
                except Exception as e:
                    error_msg = f"DEBUG (PID {multiprocessing.current_process().pid}): Error running test case with inputs {inputs}: {str(e)}"  # Modified print
                    if not self.quiet:
                        print(error_msg)
                    return False, []  # Existing return

            if not self.quiet:
                print(
                    f"DEBUG (PID {multiprocessing.current_process().pid}): Finished all test cases successfully.")  # Modified print
            return True, results
        except Exception as e:
            error_msg = f"DEBUG (PID {multiprocessing.current_process().pid}): Error in run_test_cases outer try: {str(e)}"  # Modified print
            if not self.quiet:
                print(error_msg)
            return False, []
        finally:
            if not self.quiet:
                print(f"DEBUG (PID {multiprocessing.current_process().pid}): Exiting run_test_cases")  # Modified print
    
    def _compare_outputs(self, std_output: Any, att_output: Any, tolerance: float = None) -> bool:
        """Compare two outputs of potentially different types.

        Args:
            std_output: Standard (expected) output
            att_output: Attempt (actual) output
            tolerance: Absolute tolerance for numerical comparisons. If None, uses default tolerance of 1e-6

        Returns:
            bool: True if outputs match within tolerance
        """
        # Set default tolerance if none provided
        default_tolerance = 1e-6
        tolerance = tolerance if tolerance is not None else default_tolerance

        # Helper to normalize complex numbers to standard form (x + yj, always showing real part)
        def normalize_complex(c):
            """Normalize complex number to always show real part, even if zero.
            This ensures consistent comparison: 1j -> 0+1j, -1j -> 0-1j"""
            if isinstance(c, (complex, np.complex64, np.complex128)):
                c = complex(c)
                # Return as complex (Python already treats 1j as 0+1j internally)
                return c
            return c

        # Helper to convert string to number if possible
        def try_convert(val):
            if isinstance(val, str):
                # Cleanup string
                val_clean = val.strip().replace(' ', '')
                # Try complex first (look for j or i)
                if 'j' in val_clean or 'i' in val_clean:
                    try:
                        # Replace i with j for python complex parsing
                        val_clean_j = val_clean.replace('i', 'j')
                        # Remove parentheses if present (e.g., "(1+2j)" -> "1+2j", "(-0-1j)" -> "-0-1j")
                        val_clean_j = val_clean_j.replace('(', '').replace(')', '')
                        # Handle special cases like "-0-1j" which should be parsed as "-1j"
                        # Python's complex() can handle "-0-1j" but let's be explicit
                        if val_clean_j.startswith('-0-') or val_clean_j.startswith('+0-'):
                            # Replace "-0-" with "-" and "+0-" with "-"
                            val_clean_j = val_clean_j.replace('-0-', '-').replace('+0-', '-')
                        if val_clean_j.startswith('-0+') or val_clean_j.startswith('+0+'):
                            # Replace "-0+" with "+" and "+0+" with "+"
                            val_clean_j = val_clean_j.replace('-0+', '+').replace('+0+', '+')
                        # Parse as complex
                        c = complex(val_clean_j)
                        # Normalize to ensure consistent representation
                        return normalize_complex(c)
                    except:
                        pass
                # Try float
                try:
                    return float(val_clean)
                except:
                    pass
            return val

        # 1. Normalize containers: Convert tuples to lists for comparison
        if isinstance(std_output, tuple):
            std_output = list(std_output)
        if isinstance(att_output, tuple):
            att_output = list(att_output)

        # 2. Handle Arrays
        if isinstance(std_output, np.ndarray) or isinstance(att_output, np.ndarray):
            try:
                # If one is not array, try to convert (e.g. list to array)
                std_arr = np.array(std_output) if not isinstance(std_output, np.ndarray) else std_output
                att_arr = np.array(att_output) if not isinstance(att_output, np.ndarray) else att_output

                if std_arr.shape != att_arr.shape:
                    return False

                # Check for boolean arrays specifically?
                if std_arr.dtype.kind == 'b' or att_arr.dtype.kind == 'b':
                    return np.all(std_arr == att_arr)

                # Numeric arrays (float/int/complex)
                # Use np.isclose or abs diff
                return np.all(np.abs(std_arr - att_arr) <= tolerance)
            except:
                pass  # Fallback to recursive list comparison if array conversion/op fails

        # 3. Handle Lists (recursive)
        if isinstance(std_output, list) and isinstance(att_output, list):
            if len(std_output) != len(att_output):
                return False
            # For each pair, try to convert strings to numbers before recursive comparison
            pairs = []
            for s, a in zip(std_output, att_output):
                # Convert strings to numbers if possible, but preserve original if conversion fails
                s_conv = try_convert(s) if isinstance(s, str) else s
                a_conv = try_convert(a) if isinstance(a, str) else a
                # Normalize complex numbers
                s_conv = normalize_complex(s_conv) if isinstance(s_conv, (complex, np.complex64, np.complex128)) else s_conv
                a_conv = normalize_complex(a_conv) if isinstance(a_conv, (complex, np.complex64, np.complex128)) else a_conv
                pairs.append((s_conv, a_conv))
            return all(self._compare_outputs(s, a, tolerance) for s, a in pairs)

        # 4. Handle Dicts
        if isinstance(std_output, dict) and isinstance(att_output, dict):
            if std_output.keys() != att_output.keys():
                return False
            return all(self._compare_outputs(std_output[k], att_output[k], tolerance)
                       for k in std_output)

        # 5. Handle Scalars (Numeric & Complex)
        # Try converting strings
        std_val = try_convert(std_output)
        att_val = try_convert(att_output)

        # Normalize complex numbers before comparison
        std_val = normalize_complex(std_val) if isinstance(std_val, (complex, np.complex64, np.complex128)) else std_val
        att_val = normalize_complex(att_val) if isinstance(att_val, (complex, np.complex64, np.complex128)) else att_val

        # Check for Complex
        is_complex_std = isinstance(std_val, (complex, np.complex64, np.complex128))
        is_complex_att = isinstance(att_val, (complex, np.complex64, np.complex128))

        if is_complex_std or is_complex_att:
            try:
                # Convert both to complex and compare with tolerance
                std_c = complex(std_val)
                att_c = complex(att_val)
                return abs(std_c - att_c) <= tolerance
            except:
                return False

        # Check for Real Numbers (including bools)
        def is_number(x):
            return isinstance(x, (int, float, np.number, bool, np.bool_))

        if is_number(std_val) and is_number(att_val):
            try:
                return abs(float(std_val) - float(att_val)) <= tolerance
            except:
                return False

        # 6. Sympy / other objects
        if hasattr(std_output, '__add__'):  # For sympy Add type
            try:
                diff = std_output - att_output
                if hasattr(diff, 'is_zero'):
                    return diff.is_zero
                # Fallback if subtraction works but no is_zero (unlikely for Sympy)
            except:
                pass

        # 7. Final Fallback
            return std_output == att_output

    def _grade_solution(self, attempt: Dict, problem: Dict, solution_generating_model: str = None) -> Dict:
        if not self.quiet:
            print("DEBUG: Entering _grade_solution (Main Process)")

        final_result = {  # Default error result
            "verified": 0,
            "error": "Grading failed unexpectedly before attempt loop.",
            "test_cases": []
        }

        try:
            # Extract code from both solutions
            if not self.quiet:
                print("DEBUG: Extracting standard code...")
            standard_code = self._extract_python_code(problem["problem_details"]["Code"])
            if not self.quiet:
                print(f"DEBUG: Standard code extracted: {bool(standard_code)}")
                print("DEBUG: Extracting attempt codes...")
            attempt_codes = self._extract_attempt_codes(attempt["detailed_solution"])
            if not self.quiet:
                print(f"DEBUG: Attempt codes extracted: {len(attempt_codes)} blocks found.")

            if not standard_code:  # Only need standard code here, attempt code checked later
                if not self.quiet:
                    print("DEBUG: Missing standard code, returning error.")
                return {
                    "verified": 0,
                    "error": "Could not extract standard code from problem",
                    "details": {"standard_code_found": False}
                }

            if not attempt_codes:
                if not self.quiet:
                    print("DEBUG: Missing attempt code, returning error.")
                return {
                    "verified": 0,
                    "error": "Could not extract any code blocks from attempt solution",
                    "details": {"attempt_code_found": False}
                }

            # Get test cases from problem details
            if "test" not in problem["problem_details"]:
                print("DEBUG: No test cases found, returning error.")
                return {"verified": 0, "error": "No test cases found in problem details"}
            test = problem["problem_details"]["test"]

            # --- Standard Function Execution (run in main process, assumed safe) ---
            if not self.quiet:
                print("DEBUG: Creating standard function...")
            standard_func = self.create_function(standard_code)  # Use self.create_function
            if not self.quiet:
                print(f"DEBUG: Standard function created: {bool(standard_func)}")
            if not standard_func:
                if not self.quiet:
                    print("DEBUG: Failed to create standard function, returning error.")
                # Use self.create_function's error print
                return {"verified": 0, "error": "Could not create standard function"}

            if not self.quiet:
                print("DEBUG: Running test cases for standard function...")
            # Use self.run_test_cases, keep its timeout as first defense
            standard_success, standard_results = self.run_test_cases(test, standard_func)
            if not self.quiet:
                print(f"DEBUG: Standard test cases success: {standard_success}")
            if not standard_success:
                if not self.quiet:
                    print("DEBUG: Error running standard test cases, returning error.")
                # Use self.run_test_cases's error print
                return {"verified": 0, "error": "Error running standard test cases"}
            # --- End Standard Function Execution ---

            if not self.quiet:
                print("DEBUG: Starting loop through attempt codes...")
            # Try each attempt code block using multiprocessing
            for idx, attempt_code in enumerate(attempt_codes):
                if not self.quiet:
                    print(f"DEBUG: Processing attempt code block {idx + 1}/{len(attempt_codes)} in a subprocess...")

                result_queue = multiprocessing.Queue()
                process = multiprocessing.Process(
                    target=_execute_grading_task,
                    args=(attempt_code, test, standard_results, self.timeout_seconds, result_queue, self.quiet)
                )

                process.start()

                try:
                    # Wait for the result from the subprocess with timeout
                    # Add a small buffer (e.g., 2 seconds) to the process timeout
                    if not self.quiet:
                        print(f"DEBUG: Waiting for subprocess result (timeout: {self.timeout_seconds + 2}s)...")
                    result = result_queue.get(timeout=self.timeout_seconds + 2)
                    if not self.quiet:
                        print("DEBUG: Subprocess result received.")

                    # Check if the result indicates success
                    if result.get("verified") == 1:
                        if not self.quiet:
                            print("DEBUG: Found matching solution via subprocess, returning success.")
                        final_result = result  # Store the successful result
                        process.join()  # Wait for process to finish cleanly
                        return final_result  # Return immediately upon success

                    else:
                        # Store the error/non-match result from this attempt, but continue to next attempt
                        final_result = result
                        if not self.quiet:
                            print(f"DEBUG: Attempt {idx + 1} did not verify: {result.get('error', 'Output mismatch')}")

                except Empty:
                    # Timeout occurred waiting for the queue
                    if not self.quiet:
                        print(f"DEBUG: Timeout waiting for subprocess for attempt block {idx + 1}. Terminating process.")
                    process.terminate()  # Forcefully terminate the stuck process
                    final_result = {  # Set result to indicate timeout
                        "verified": 0,
                        "error": f"Grading timed out after {self.timeout_seconds + 2} seconds for attempt block {idx + 1}",
                        "test_cases": []
                    }
                    # Don't return yet, maybe a later block works, but store the timeout error

                except Exception as e:
                    if not self.quiet:
                        print(f"DEBUG: Error managing subprocess for attempt block {idx + 1}: {str(e)}")
                    if process.is_alive():
                        process.terminate()
                    final_result = {
                        "verified": 0,
                        "error": f"Error managing subprocess: {str(e)}",
                        "test_cases": []
                    }
                    # Don't return yet, store the error

                finally:
                    # Ensure the process is joined (waited for) if it hasn't been terminated
                    if process.is_alive():
                        if not self.quiet:
                            print(f"DEBUG: Joining process {process.pid} for attempt {idx + 1}...")
                        process.join(timeout=1)  # Short timeout for join
                        if process.is_alive():
                            if not self.quiet:
                                print(
                                    f"WARN: Process {process.pid} did not join cleanly after result/timeout, terminating again.")
                            process.terminate()
                        else:
                            if not self.quiet:
                                print(f"DEBUG: Process {process.pid} joined cleanly.")
                    else:
                        if not self.quiet:
                            print(f"DEBUG: Process for attempt {idx + 1} already terminated or finished.")
                    result_queue.close()  # Close the queue

            # If loop finishes without returning a verified=1 result
            if not self.quiet:
                print("DEBUG: No valid solution found among all attempt code blocks.")
            # final_result will contain the outcome of the last processed block (error, timeout, or non-match)
            return final_result

        except Exception as e:
            if not self.quiet:
                print(f"DEBUG: Unexpected exception in _grade_solution (Main Process): {str(e)}")
            # Return a generic error if something unexpected happens outside the loop
            return {
                "verified": 0,
                "error": f"Unexpected error in main grading logic: {str(e)}",
                "test_cases": []
            }
        finally:
            if not self.quiet:
                print("DEBUG: Exiting _grade_solution (Main Process)")

    def grade_all_problems(self, regrade_all: bool = False) -> None:
        problem_files = self.get_problem_files()
        for problem_file in tqdm(problem_files):
            problem_path = self.problems_dir / problem_file
            try:
                self._grade_problem(
                    problem_path,
                    regrade_all
                )
            except Exception as e:
                print(f"Error grading problem {problem_path}: {str(e)}")
                print(f"Skipping this problem and continuing with others...")
                continue

