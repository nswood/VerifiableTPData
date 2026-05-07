from datetime import datetime
import simplejson as json
from pathlib import Path
import re
from typing import Dict, List, Optional
from collections import defaultdict
import textwrap
import logging

from .problem_processor_base import load_problem_with_lock


def clean_cot_for_latex(text: str) -> str:
    """
    Clean and format Chain-of-Thought content for LaTeX compilation.

    This function handles common formatting issues in model-generated CoT content:
    - Escapes special LaTeX characters outside of math mode
    - Converts markdown code blocks to LaTeX format
    - Handles arrows and other special sequences
    - Preserves already-valid LaTeX content

    Args:
        text: The raw CoT text from model output

    Returns:
        LaTeX-safe formatted text
    """
    if not text:
        return text

    # First, protect existing LaTeX constructs by replacing them with placeholders
    placeholders = {}
    placeholder_counter = [0]

    def make_placeholder(match, prefix="PROTECTED"):
        """Create a unique placeholder for protected content."""
        key = f"___{prefix}_{placeholder_counter[0]}___"
        placeholder_counter[0] += 1
        placeholders[key] = match.group(0)
        return key

    # Protect display math environments ($$...$$ and \[...\])
    text = re.sub(r'\$\$(.+?)\$\$', lambda m: make_placeholder(m, "DISPMATH"), text, flags=re.DOTALL)
    text = re.sub(r'\\\[(.+?)\\\]', lambda m: make_placeholder(m, "DISPMATH2"), text, flags=re.DOTALL)

    # Protect inline math ($...$) - be careful not to match currency
    text = re.sub(r'(?<!\$)\$(?!\$)([^\$\n]+?)(?<!\$)\$(?!\$)', lambda m: make_placeholder(m, "INLINEMATH"), text)

    # Protect \begin{...}...\end{...} environments
    text = re.sub(r'\\begin\{(\w+)\}(.+?)\\end\{\1\}', lambda m: make_placeholder(m, "ENV"), text, flags=re.DOTALL)

    # Protect LaTeX commands that might contain special chars (e.g., \frac{}{}, \text{}, etc.)
    text = re.sub(r'\\(?:frac|sqrt|text|textbf|textit|mathrm|mathbf|mathcal|vec|hat|bar|tilde|dot|ddot)\{[^}]*\}(?:\{[^}]*\})?',
                  lambda m: make_placeholder(m, "CMD"), text)

    # Protect already-escaped characters
    text = re.sub(r'\\([_&%#\${}^~])', lambda m: make_placeholder(m, "ESC"), text)

    # Convert markdown code blocks to LaTeX verbatim/python environments
    def convert_code_block(match):
        lang = match.group(1) or ""
        code = match.group(2)
        if lang.lower() == "python":
            return f"\\begin{{python}}\n{code}\n\\end{{python}}"
        else:
            # Use verbatim for other languages
            return f"\\begin{{verbatim}}\n{code}\n\\end{{verbatim}}"

    text = re.sub(r'```(\w*)\s*\n?(.*?)```', convert_code_block, text, flags=re.DOTALL)

    # Handle single backtick inline code
    text = re.sub(r'`([^`]+)`', r'\\texttt{\1}', text)

    # Convert common arrow patterns to LaTeX (outside of math mode)
    arrow_replacements = [
        (r'(?<![<\-])-->(?![>\-])', r'$\\rightarrow$'),
        (r'(?<![<\-])<--(?![>\-])', r'$\\leftarrow$'),
        (r'(?<![<\-])->(?![>\-])', r'$\\rightarrow$'),
        (r'(?<![<\-])<-(?![>\-])', r'$\\leftarrow$'),
        (r'(?<![=<])=>(?![>=])', r'$\\Rightarrow$'),
        (r'(?<![=<])<=(?![>=])', r'$\\leq$'),  # Could be ≤ or ⇐ depending on context
        (r'(?<![<])>=(?![>])', r'$\\geq$'),
        (r'(?<![<])!=(?![>])', r'$\\neq$'),
        (r'<->(?![>])', r'$\\leftrightarrow$'),
        (r'(?<![<])<=>', r'$\\Leftrightarrow$'),
    ]

    for pattern, replacement in arrow_replacements:
        text = re.sub(pattern, replacement, text)

    # Handle common physics notation patterns that should be in math mode
    # Pattern: short variable followed by ^ or _ with short subscript (e.g., phi^a, x_1)
    # But NOT long variable names like function_name which are code identifiers
    def mathify_subscript_superscript(match):
        """Convert bare subscript/superscript notation to math mode."""
        prefix = match.group(1)  # The variable/word before ^ or _
        op = match.group(2)      # ^ or _
        suffix = match.group(3)  # What comes after
        return f"${prefix}{op}{{{suffix}}}$"

    # Common physics/math variables that should be in math mode
    physics_vars = r'(?:phi|psi|chi|theta|alpha|beta|gamma|delta|epsilon|zeta|eta|kappa|lambda|mu|nu|xi|rho|sigma|tau|omega|Phi|Psi|Chi|Theta|Alpha|Beta|Gamma|Delta|Epsilon|Lambda|Omega|[a-zA-Z])'

    # Match patterns like: short_physics_var^letter or short_var_letter (1-2 chars)
    # Superscripts (^) - always mathify these as they're clearly math notation
    text = re.sub(rf'\b({physics_vars})(\^)([A-Za-z0-9]+)\b(?![}}\$])', mathify_subscript_superscript, text)

    # Subscripts (_) - only mathify for short variable names (likely physics notation)
    # Long names like "function_name" should just have underscores escaped
    text = re.sub(rf'\b({physics_vars})(_)([A-Za-z0-9]{{1,3}})\b(?![}}\$])', mathify_subscript_superscript, text)

    # Handle Greek letters followed by subscripts/superscripts
    greek_letters = r'(?:alpha|beta|gamma|delta|epsilon|zeta|eta|theta|iota|kappa|lambda|mu|nu|xi|omicron|pi|rho|sigma|tau|upsilon|phi|chi|psi|omega|Gamma|Delta|Theta|Lambda|Xi|Pi|Sigma|Upsilon|Phi|Psi|Omega)'
    text = re.sub(rf'\\({greek_letters})(\^|_)([A-Za-z0-9]+)\b(?![}}\$])',
                  lambda m: f"$\\{m.group(1)}{m.group(2)}{{{m.group(3)}}}$", text)

    # Escape special LaTeX characters that aren't already handled
    # Order matters: & before others since it's commonly used
    special_chars = [
        # (pattern, replacement) - escape if not already escaped
        (r'(?<!\\)&(?!amp;)', r'\\&'),
        (r'(?<!\\)%', r'\\%'),
        (r'(?<!\\)#', r'\\#'),
        # Don't escape $ here as it might be math delimiter
        # Don't escape _ and ^ here as they might be in math mode
    ]

    for pattern, replacement in special_chars:
        text = re.sub(pattern, replacement, text)

    # Handle bare underscores and carets that are NOT in a math context
    # This is tricky - we want to escape standalone ones but not math ones
    # Strategy: escape _ and ^ only when they appear in clearly non-math context
    # e.g., variable_name should become variable\_name, but x_1 should be $x_1$

    # Escape underscores in snake_case identifiers (common in code)
    text = re.sub(r'([a-z])_([a-z])', r'\1\\_\2', text, flags=re.IGNORECASE)

    # Handle remaining bare underscores at word boundaries
    text = re.sub(r'(?<=[a-zA-Z0-9])_(?=[a-zA-Z0-9])', r'\\_', text)

    # Clean up any double-escaped characters
    text = re.sub(r'\\\\([_&%#])', r'\\\1', text)

    # Restore all placeholders
    for key, value in placeholders.items():
        text = text.replace(key, value)

    # Final cleanup: fix common issues
    # Remove any remaining raw underscores/carets that would break LaTeX
    # but be conservative to avoid breaking valid content

    return text


def break_lines(match):
    content = match.group(1)
    return break_lines_cleaned(content)


def break_lines_cleaned(content: str) -> str:
    """
    Wrap think content in verbatim environment with line breaking.

    Args:
        content: The content inside <think> tags

    Returns:
        LaTeX verbatim-wrapped content
    """
    # Split content by original line breaks and process line by line
    new_lines = []
    for line in content.splitlines():
        # Keep empty lines; otherwise use textwrap.wrap for auto line wrapping
        if line.strip():
            wrapped = textwrap.wrap(line, width=85)
            new_lines.append('\n'.join(wrapped))
        else:
            new_lines.append('')
    new_content = '\n'.join(new_lines)
    return r'\begin{verbatim}' + '<think>' + new_content + '\n</think>' + r'\end{verbatim}'


class LaTeXGenerator:
    def __init__(
        self,
        problems_dir: str = "",
        problem_path: str = None,
        analysis_only: bool = False,
        visualize_problem_only: bool = False,
        attempts_included_per_model: int = None,
        baseline_model: str = "chatgpt-4o-latest",
        selected_solvers: list = None,
        output_name: str = None,
        show_verification: bool = True,  # Whether to show auto-verification results
        show_cot_r1: bool = False,  # Whether to show DeepSeek-R1 CoT `<think>` content
        only_compile_correct_attempts: bool = False,
        best_of_n: int = 5,
        show_model_performance_by_problem: bool = False,
        include_topics: list = None,  # Only include problems with these Topic Entry IDs
        exclude_topics: list = None,  # Exclude problems with these Topic Entry IDs
    ):
        # Get the absolute path of the project root directory
        self.project_root = Path(__file__).parent.parent

        # Use absolute path for physics_problems directory
        self.problems_dir = self.project_root / problems_dir
        self.problem_path = problem_path
        self.visualize_problem_only = visualize_problem_only

        self.analysis_only = analysis_only
        self.baseline_model = baseline_model

        self.attempts_included_per_model = attempts_included_per_model
        if not self.problems_dir.exists():
            raise FileNotFoundError(f"Problems directory not found: {self.problems_dir}")

        self.selected_solvers = selected_solvers
        self.output_name = output_name
        self.show_verification = show_verification
        self.show_cot_r1 = show_cot_r1
        self.only_compile_correct_attempts = only_compile_correct_attempts

        # Configure best-of-N, capping at attempts_included_per_model when provided
        self.best_of_n = best_of_n if best_of_n is not None else 5
        if self.attempts_included_per_model is not None and self.best_of_n is not None:
            self.best_of_n = min(self.best_of_n, self.attempts_included_per_model)
        self.show_model_performance_by_problem = show_model_performance_by_problem
        self.include_topics = include_topics
        self.exclude_topics = exclude_topics
        self.logger = logging.getLogger(__name__)

    def _load_all_problems(self):
        """Load all problems from individual files"""
        problems = []
        if self.problem_path:
            problem_file = Path(self.problem_path)
            if not problem_file.is_absolute():
                problem_file = self.project_root / problem_file
            if problem_file.exists():
                problem = load_problem_with_lock(problem_file)
                # Apply topic filtering even for single problem
                if self._should_include_problem(problem):
                    problems.append(problem)
            else:
                logging.warning(f"Problem file not found: {problem_file}")
            return problems

        for problem_file in sorted(self.problems_dir.rglob("*.json")):
            problem = load_problem_with_lock(problem_file)
            # Apply topic filtering
            if self._should_include_problem(problem):
                problems.append(problem)
        self.logger.info(f"Loaded {len(problems)} problems")
        return problems

    def _should_include_problem(self, problem: Dict) -> bool:
        """Check if a problem should be included based on topic filters."""
        # Some directories contain auxiliary JSONs (assignment.json, quality_report.json)
        # that don't follow the problem schema; skip them silently.
        if not isinstance(problem, dict) or "problem_metadata" not in problem:
            return False
        topic_id = problem.get("problem_metadata", {}).get("Topic Entry ID", "")

        # If include_topics is set, only include problems with matching topic IDs
        if self.include_topics and topic_id not in self.include_topics:
            return False

        # If exclude_topics is set, exclude problems with matching topic IDs
        if self.exclude_topics and topic_id in self.exclude_topics:
            return False

        return True

    def generate_latex(self, output_path: str = None):
        """Generate the LaTeX report"""
        # Generate default output path with timestamp if none provided
        if output_path is None:
            if self.problem_path and not self.output_name:
                self.output_name = self.problem_path.split("/")[-1].replace(".json", "")
            if self.output_name:
                output_path = self.project_root / "output" / f"{self.output_name}.tex"
            else:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_path = self.project_root / "output" / f"solutions_report_{timestamp}.tex"

            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

        # Build the document content
        content = []

        content.extend([
            "\\documentclass{article}",
            "\\usepackage{amsmath}",
            "\\usepackage{amsfonts}",
            "\\usepackage{amsthm}",
            "\\usepackage{graphicx}",
            "\\usepackage[margin=1in]{geometry}",
            "\\usepackage{pgfplots}",
            "\\usepackage{xcolor}",
            "\\usepackage{tcolorbox}",
            "\\usepackage[utf8]{inputenc}",
            "\\usepackage{wasysym}",
            "\\usepackage{mathtools}",
            "\\usepackage{listings}",
            "\\usepackage{hyperref} ",
            "% Inline python environment (replaces pythonhighlight package)",
            "\\lstnewenvironment{python}[1][]{\\lstset{language=Python,basicstyle=\\ttfamily\\small,breaklines=true,keywordstyle=\\color{blue},commentstyle=\\color{gray},stringstyle=\\color{red},showstringspaces=false,#1}}{}",
            "\\usepackage{bm}",
            "\\usepackage{slashed}",
            "\\usepackage{cancel}",
            "\\tcbuselibrary{breakable}",
            "",
            "\\newtheorem{conjecture}{Conjecture}",
            "\\newtheorem{theorem}{Theorem}",
            "\\newtheorem{proposition}{Proposition}",
            "\\newtheorem{claim}{Claim}",
            "\\newtheorem{lemma}{Lemma}",
            "\\newtheorem{corollary}{Corollary}",
            "\\newtheorem{definition}{Definition}",
            "",
        ])


        # Add document class and preamble with tcolorbox settings
        content.extend([
            "% Define colors for different solutions",
            "\\definecolor{standardcolor}{RGB}{255, 243, 205}  % Light yellow",
            "\\definecolor{attemptcolor}{RGB}{240, 240, 240}  % Light gray",
            "\\definecolor{problemcolor}{RGB}{230, 245, 255}  % Light blue",
            "\\definecolor{verifiedcolor}{RGB}{200, 255, 200}  % Light green for verified solutions",
            "\\definecolor{notverifiedcolor}{RGB}{255, 200, 200}  % Light red for non-verified solutions",
            "",
            "% Define tcolorbox styles",
            "\\tcbset{",
            "    standard/.style={",
            "        colback=standardcolor,",
            "        colframe=gray!50,",
            "        boxrule=0.5mm,",
            "        breakable,",
            "        sharp corners,",
            "        before skip=10pt,",
            "        after skip=10pt",
            "    },",
            "    attempt/.style={",
            "        colback=attemptcolor,",
            "        colframe=gray!50,",
            "        boxrule=0.5mm,",
            "        breakable,",
            "        sharp corners,",
            "        before skip=10pt,",
            "        after skip=10pt",
            "    },",
            "    problem/.style={",
            "        colback=problemcolor,",
            "        colframe=blue!30,",
            "        boxrule=0.5mm,",
            "        breakable,",
            "        sharp corners,",
            "        before skip=10pt,",
            "        after skip=10pt",
            "    }",
            "}",
            "",
            "\\begin{document}",
            "",
        ])

        if self.problem_path:
            problem_name = self.problem_path.split("/")[-1].replace(".json", "")
            escaped_problem_name = problem_name.replace('_', r'\_')
            content.append(f"\\title{{{escaped_problem_name} Report}}")
        elif self.visualize_problem_only:
            content.append(f"\\title{{Problem Set Visualization}}")
        elif self.output_name:
            # Use custom output name for title
            escaped_output_name = self.output_name.replace('_', r'\_')
            content.append(f"\\title{{{escaped_output_name}}}")
        else:
            # Truncate problems_dir for a cleaner title
            # Get absolute path to handle relative paths correctly
            abs_problems_dir = self.problems_dir.resolve()
            # Take the last 3 parts of the path for the title
            truncated_path = "/".join(abs_problems_dir.parts[-3:])
            escaped_path = truncated_path.replace('_', r'\_')
            content.append(f"\\title{{{escaped_path} - Problem Report}}")

        content.extend([
            f"\\date{{Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}}}",
            "\\author{TPBench.org}",
            "\\maketitle",
            "",
        ])
        # Only include table of contents when generating multi-problem reports
        if not self.problem_path:
            content.append("\\tableofcontents")
            content.append("\\newpage")


        # Add grade distribution analysis
        problems = self._load_all_problems()
        content.extend(self._generate_grade_analysis(problems))

        if self.visualize_problem_only:
            for problem in problems:
                content.extend(self._format_problem(problem))
            content.append("\\end{document}")
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(content))
            return output_path
        

        if not self.analysis_only:
            content.append("\\newpage")
            # Process each problem
            for problem in problems:
                content.extend(self._format_problem(problem))

        # Close document
        content.append("\\end{document}")

        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(content))
        return output_path

    def _get_verifier_results(self, attempt):
        """
        Get verifier_result from attempt, handling both dict (new format) and list (old format).
        Returns a list of results for backward compatibility.
        """
        verifier_result = attempt.get('verifier_result')
        if not verifier_result:
            return []
        if isinstance(verifier_result, dict):
            return [verifier_result]
        elif isinstance(verifier_result, list):
            return verifier_result
        else:
            return []

    def _format_problem(self, problem):
        """Format a single problem for LaTeX"""
        elements = []
        
        # Problem header and text wrapped in tcolorbox
        if self.problem_path:
            problem_name = self.problem_path.split("/")[-1].replace(".json", "")
        else:
            problem_name = problem['problem_id']
        escaped_problem_name = problem_name.replace('_', r'\_')
        elements.extend([
            f"\\section{{Problem {escaped_problem_name}, Difficulty level: {problem.get('problem_metadata', {}).get('Difficulty level', 'Unknown')}}}",
            "",
            "\\begin{tcolorbox}[problem]",
            "\\textbf{Problem Text:}\\",
            self._process_latex_text(problem['problem_details']['Problem Statement']),
            "\\end{tcolorbox}",
            ""
        ])

        # Standard solution wrapped in tcolorbox
        standard_solution = [
            "\\subsection{{Expert Solution}}",
            "",
            "\\begin{tcolorbox}[standard]",
            "\\textbf{Detailed Steps:}",
            self._process_latex_text(problem['problem_details']['Solution']),
            "",
        ]

        if 'Answer' in problem['problem_details'] and problem['problem_details']['Answer'].strip():
            standard_solution.extend([
                "\\textbf{Final Answer:}",
                self._process_latex_text(problem['problem_details']['Answer']),
                "",
            ])

        # Add Answer Requirements if present
        if 'Answer Requirements' in problem['problem_details'] and problem['problem_details']['Answer Requirements'].strip():
            standard_solution.extend([
                "\\textbf{Answer Requirements:}",
                self._process_latex_text(problem['problem_details']['Answer Requirements']),
                "",
            ])

        # Add Code if present
        if 'Code' in problem['problem_details'] and problem['problem_details']['Code'].strip():
            standard_solution.extend([
                "\\textbf{Code Implementation:}",
                self._process_latex_text(problem['problem_details']['Code']),
                "",
            ])

        standard_solution.extend([
            "\\end{tcolorbox}",
            "\\newpage",
            ""
        ])

        elements.extend(standard_solution)

        if self.visualize_problem_only:
            return elements

        if self.only_compile_correct_attempts:
            exist_correct_attempt = False

        # Model solutions
        if 'model_solutions' in problem and problem['model_solutions']:
            elements.append("\\subsection{{Model Solutions}}")

            # Filter and sort model_solutions by 'model' name
            filtered_solutions = [
                sol for sol in problem['model_solutions']
                if not self.selected_solvers or sol['model'] in self.selected_solvers
            ]

            metric = self.calculate_verification_metrics([problem])
            model_accuracy = {}
            for key, value in metric.items():
                model_accuracy[key] = value['success_rate']
            self.logger.debug(f"Model accuracy map: {model_accuracy}")

            for model_solution in sorted(
                filtered_solutions,
                key=lambda x: model_accuracy.get(x['model'], 0),
                reverse=True,
            ):
                elements.append(f"\\subsubsection{{Model: {model_solution['model']}}}")
                elements.append("")

                # Format each attempt
                for attempt_idx, attempt in enumerate(
                    model_solution['attempts'][: self.attempts_included_per_model]
                ):
                    # Get verifier results once per attempt
                    verifier_results = self._get_verifier_results(attempt)

                    if self.only_compile_correct_attempts:
                        if not verifier_results or verifier_results[0].get('verified', 0) != 1:
                            continue
                        exist_correct_attempt = True
                    # Remove any \documentclass and \end{document} from the solution text
                    detailed_solution = attempt['detailed_solution']
                    detailed_solution = re.sub(r'\\documentclass.*?\n', '', detailed_solution)
                    detailed_solution = re.sub(r'\\end\{document\}', '', detailed_solution)

                    attempt_solution = [
                        f"\\paragraph{{Attempt {attempt_idx + 1}}} ({attempt['timestamp']})",
                        "",
                        "\\begin{tcolorbox}[attempt]",
                        "\\textbf{Detailed Solution:}",
                        self._process_latex_text(detailed_solution, is_cot=True),
                        "",
                    ]

                    # Modify the verification results section to check show_verification flag
                    verifier_results = self._get_verifier_results(attempt)
                    if verifier_results and self.show_verification:
                        attempt_solution.append("\\textbf{Verification Results:}")
                        for result in verifier_results:
                            verification_status = result.get('verified', 0)
                            timestamp = result.get('timestamp', '')
                            grading_model = result.get('grading_model', '')
                            
                            # Create colored box based on verification status
                            if verification_status == 1:
                                status_box = "\\colorbox{green!20}{\\textbf{Correct}}"
                            elif verification_status == 0:
                                status_box = "\\colorbox{red!20}{\\textbf{Incorrect}}"
                            else:
                                status_box = "\\colorbox{yellow!20}{\\textbf{Unknown}}"
                            
                            attempt_solution.extend([
                                f"\\subparagraph{{Auto verification result}} ({timestamp})",
                                f"Status: {status_box}\\\\",
                            ])
                            
                            # Add error message if verification failed
                            # if 'error' in result and result['error']:
                            #     attempt_solution.append(
                            #         f"Error: {self._process_latex_text(result['error'])}\\\\",
                            #     )
                            attempt_solution.append("")

                    attempt_solution.extend([
                        "\\end{tcolorbox}",
                        ""
                    ])
                    elements.extend(attempt_solution)
                elements.append("\\newpage")
        if self.only_compile_correct_attempts and not exist_correct_attempt:
            return []
        return elements

    def _extract_python_code(self, text: str) -> Optional[str]:
        # Try \\begin{python} format first
        pattern1 = r"\\begin{python}(.*?)\\end{python}"
        matches = re.findall(pattern1, text, re.DOTALL)
        if matches:
            extracted_code = matches[0].strip()
            return self._wrap_with_latex(extracted_code)

        # Try ```python format next
        pattern2 = r"```python\s*(.*?)```"
        matches = re.findall(pattern2, text, re.DOTALL)
        if matches:
            extracted_code = matches[0].strip()
            return self._wrap_with_latex(extracted_code)
        return None

    def _wrap_with_latex(self, code: str) -> str:
        return f"""
        \\begin{{python}}
        {code}
        \\end{{python}}
        """

    def _process_latex_text(self, text: str, is_cot: bool = False) -> str:
        """
        Process text for LaTeX output, handling special characters and formatting.

        Args:
            text: The text to process
            is_cot: Whether this is Chain-of-Thought content that needs extra cleaning
        """
        # Apply CoT cleaning for model-generated content
        if is_cot:
            text = clean_cot_for_latex(text)

        # Add special handling for DeepSeek-R1 model's <think> tags
        if "<think>" in text and "</think>" in text:
            if self.show_cot_r1:
                # Extract and clean the think content, then wrap in verbatim
                def clean_and_wrap_think(match):
                    think_content = match.group(1)
                    # Apply cleaning to think content
                    think_content = clean_cot_for_latex(think_content)
                    return break_lines_cleaned(think_content)
                text = re.sub(r'<think>(.*?)</think>', clean_and_wrap_think, text, flags=re.DOTALL)
            else:
                text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)

        # Remove any LaTeX document structure commands
        text = re.sub(r'\\documentclass.*?\n', '', text)
        text = re.sub(r'\\begin\{document\}', '', text)
        text = re.sub(r'\\end\{document\}', '', text)
        text = re.sub(r'\\usepackage.*?\n', '', text)

        # Replace Python code blocks with formatted LaTeX
        pattern1 = r"\\begin{python}(.*?)\\end{python}"
        text = re.sub(pattern1, lambda m: self._wrap_with_latex(m.group(1).strip()), text, flags=re.DOTALL)

        pattern2 = r"```python\s*(.*?)```"
        text = re.sub(pattern2, lambda m: self._wrap_with_latex(m.group(1).strip()), text, flags=re.DOTALL)
        text = re.sub(r'`', '', text)

        pattern1 = r"\\begin{python}(.*?)\\end{python}"
        code_blocks = re.findall(pattern1, text, re.DOTALL)
        text = re.sub(pattern1, lambda m: f"CODE_BLOCK_{code_blocks.index(m.group(1))}", text, flags=re.DOTALL)

        # Process non-code text for bold syntax
        text = re.sub(r'\*\*(.*?)\*\*', r'\\textbf{\1}', text)

        # Restore the code blocks
        for i, code_block in enumerate(code_blocks):
            placeholder = f"CODE_BLOCK_{i}"
            text = text.replace(placeholder, f"\\begin{{python}}{code_block}\\end{{python}}")
        # # Convert markdown headers to LaTeX sections
        # text = re.sub(r'^### (.*?)$', r'\\subsubsection*{\1}', text, flags=re.MULTILINE)
        # text = re.sub(r'^## (.*?)$', r'\\subsection*{\1}', text, flags=re.MULTILINE)
        # text = re.sub(r'^# (.*?)$', r'\\section*{\1}', text, flags=re.MULTILINE)

        text = re.sub(r'^#+ ', '', text, flags=re.MULTILINE)

        # Replace single $ with inline math mode
        text = re.sub(r'(?<!\$)\$(?!\$)(.*?)(?<!\$)\$(?!\$)', r'$\1$', text)

        # Replace double $$ with display math mode
        text = re.sub(r'\$\$(.*?)\$\$', r'\\[\1\\]', text)

        return text

    @staticmethod
    def _append_table_end(content, custom_message=None):
        content += ["\\hline", "\\end{tabular}", "\\end{center}", ""]
        if custom_message:
            content.append(custom_message)
        return content

    def _generate_grade_analysis(self, problems):
        """Generate verification-based analysis (no LLM grading)."""
        content = [
            "\\section{Grade Distribution Analysis}",
            "",
            "\\subsection{Auto-Verification Results}",
            "\\begin{center}",
            "\\begin{tabular}{|l|c|c|c|c|c|}",
            "\\hline",
            f"Model & Correct & Incorrect & Unknown & Success Rate & Best-of-{self.best_of_n} \\\\",
            "\\hline",
        ]

        verification_metrics = self.calculate_verification_metrics(problems)
        bestn_overall = self.calculate_best_of_n_metrics(problems, self.best_of_n)
        for model, metrics in verification_metrics.items():
            if self.selected_solvers and model not in self.selected_solvers:
                continue
            success_rate = metrics['success_rate'] * 100  # Convert to percentage
            bestn_pct = bestn_overall.get(model, 0.0) * 100
            content.append(
                f"{model} & {metrics['correct']} & {metrics['incorrect']} & "
                f"{metrics['unknown']} & {success_rate:.1f}\\% & {bestn_pct:.1f}\\% \\\\"
            )

        content = self._append_table_end(
            content,
            "\\small{Note: Success Rate = Correct / (Correct + Incorrect) × 100\\%}",
        )

        # Additional verification-based analysis only for multi-problem reports
        if not self.problem_path and self.show_model_performance_by_problem:
            # Add new table for per-problem verification results
            # Determine which two models to display: prefer o3-mini and o1 if both exist;
            # otherwise, select the first two encountered solution models (respecting selected_solvers).
            available_models = []
            for p in problems:
                for sol in p.get('model_solutions', []):
                    m = sol.get('model')
                    if self.selected_solvers and m not in self.selected_solvers:
                        continue
                    if m not in available_models:
                        available_models.append(m)

            preferred = ['o3-mini', 'o1']
            if all(pm in available_models for pm in preferred):
                models = preferred
            else:
                models = available_models[:2]

            # If fewer than two models are available, omit this section entirely
            if len(models) >= 2:
                m1, m2 = models[0], models[1]

                # Calculate per-problem verification scores for the selected models
                problem_scores = {}
                for model in models:
                    problem_scores[model] = {}
                    model_total = 0
                    valid_problems = 0

                    for problem in problems:
                        scores = []
                        if 'model_solutions' not in problem or not problem['model_solutions']:
                            continue
                        for solution in problem['model_solutions']:
                            model_name = solution['model']
                            if model_name != model:
                                continue
                            if self.selected_solvers and model_name not in self.selected_solvers:
                                continue
                            for attempt in solution['attempts'][:self.attempts_included_per_model]:
                                verifier_results = self._get_verifier_results(attempt)
                                if verifier_results:
                                    for result in verifier_results:
                                        if result.get('verified') is not None:
                                            scores.append(result.get('verified'))

                        if scores:
                            avg_score = sum(scores) / len(scores)
                            model_total += avg_score
                            valid_problems += 1
                        else:
                            avg_score = None

                        problem_scores[model][problem['problem_id']] = avg_score

                    if valid_problems > 0:
                        problem_scores[model]['avg'] = model_total / valid_problems
                    else:
                        problem_scores[model]['avg'] = 0

                # Now render the table with dynamic model names
                content.extend([
                    "\\subsection{Model Performance by Problem}",
                    "\\begin{center}",
                    "\\begin{tabular}{|l|c|c|c|}",
                    "\\hline",
                    f"Problem (Level) & {m1} & {m2} & Avg \\\\",
                    "\\hline"
                ])

                # Sort problems by difficulty level (numeric if possible, else lexicographic)
                def _level_sort_tuple(p):
                    level_val = p.get('problem_metadata', {}).get('Difficulty level', '')
                    level_str = str(level_val) if level_val is not None else ''
                    try:
                        return (0, int(level_str), p['problem_id'])
                    except Exception:
                        return (1, level_str, p['problem_id'])

                sorted_problems = sorted(problems, key=_level_sort_tuple)

                # Add rows for each problem
                for problem in sorted_problems:
                    problem_id = problem['problem_id']
                    level = problem.get('problem_metadata', {}).get('Difficulty level', '?')

                    if len(models) >= 2:
                        # Get scores for both selected models
                        s1 = problem_scores[m1].get(problem_id)
                        s2 = problem_scores[m2].get(problem_id)

                        # Format scores
                        s1_str = f"{s1:.2f}" if s1 is not None else "-"
                        s2_str = f"{s2:.2f}" if s2 is not None else "-"

                        # Calculate average of available scores
                        available_scores = [s for s in [s1, s2] if s is not None]
                        avg_str = f"{sum(available_scores) / len(available_scores):.2f}" if available_scores else "-"

                        # Add row with problem ID, level, scores
                        row = f"P{problem_id} (L{level}) & {s1_str} & {s2_str} & {avg_str} \\\\"
                        content.append(row)

                content = self._append_table_end(
                    content,
                    "\\small{Note: Values show average verification success rate (1.00 = all correct, " +
                    "0.00 = all incorrect). '-' indicates no attempts. L? indicates problem difficulty level.}"
                )

        # Per-level difficulty analysis - always shown by default for multi-problem reports
        if not self.problem_path:
            performance_metrics, level_metrics = self.calculate_model_performance_metrics(problems)

            # Show per-level performance only if any problem has a known level
            levels_present = [lvl for lvl, data in level_metrics.items() if data]
            if levels_present:
                content.extend([
                    "\\subsection{Model Performance by Problem Level}",
                ])

                def _level_key(lvl):
                    s = str(lvl) if lvl is not None else ''
                    try:
                        return (0, int(s))
                    except Exception:
                        return (1, s)

                for level in sorted(levels_present, key=_level_key):
                    content.extend([
                        f"\\subsubsection{{{str(level).title()} Level Problems}}",
                        "\\begin{center}",
                        "\\begin{tabular}{|l|c|c|c|}",
                        "\\hline",
                        f"Model & Mean Score & Best-of-{self.best_of_n} & Sample Size \\\\",
                        "\\hline"
                    ])

                    for model, metrics in level_metrics[level].items():
                        bestn = metrics.get('best_of_n', 0.0)
                        content.append(
                            f"{model} & {metrics['mean_score']:.3f} & "
                            f"{bestn:.3f} & {metrics['sample_size']} \\\\")

                    content = self._append_table_end(content)

        return content

    def calculate_model_performance_metrics(self, problems):
        # Collect all verification scores per model and problem level
        model_scores = {}
        baseline_scores = []
        # create a dictionary to store scores for each level
        level_scores = {}  # Remove predefined levels, let them be created dynamically
        level_best_successes = {}

        for problem in problems:
            problem_level = problem.get('problem_metadata', {}).get('Difficulty level', 'Unknown')
            
            # Initialize level_scores for this problem_level if it doesn't exist
            if problem_level not in level_scores:
                level_scores[problem_level] = {}
            if problem_level not in level_best_successes:
                level_best_successes[problem_level] = {}
            
            if 'model_solutions' not in problem or not problem['model_solutions']:
                continue
            for solution in problem['model_solutions']:
                model = solution['model']
                if model not in model_scores:
                    model_scores[model] = []
                if model not in level_scores[problem_level]:
                    level_scores[problem_level][model] = []
                if model not in level_best_successes[problem_level]:
                    level_best_successes[problem_level][model] = []

                # Prepare attempts for both average score and best-of-N
                attempts_all = solution['attempts']
                if self.attempts_included_per_model is not None:
                    attempts_all = attempts_all[:self.attempts_included_per_model]

                for attempt in attempts_all:
                    verifier_results = self._get_verifier_results(attempt)
                    if not verifier_results:
                        continue

                    # Calculate verification score for this attempt
                    verification_scores = []
                    for result in verifier_results:
                        verification_status = result.get('verified')
                        if verification_status is not None:  # Only count if we have a clear result
                            verification_scores.append(verification_status)

                    if verification_scores:  # Only add if we have valid scores
                        avg_score = sum(verification_scores) / len(verification_scores)
                        model_scores[model].append(avg_score)
                        level_scores[problem_level][model].append(avg_score)
                        if model == self.baseline_model:
                            baseline_scores.append(avg_score)

                # Best-of-N per problem for this model/level
                considered = attempts_all[: self.best_of_n]
                any_correct = 0
                for att in considered:
                    for vr in self._get_verifier_results(att):
                        if vr.get('verified') == 1:
                            any_correct = 1
                            break
                    if any_correct:
                        break
                if attempts_all:
                    level_best_successes[problem_level][model].append(any_correct)

        # Calculate baseline mean
        baseline_mean = sum(baseline_scores) / len(baseline_scores) if baseline_scores else 0

        # Calculate metrics per model
        metrics = {}
        for model, scores in model_scores.items():
            if not scores:
                continue

            model_mean = sum(scores) / len(scores)
            relative_performance = (model_mean / baseline_mean) if baseline_mean else 0

            metrics[model] = {
                'mean_score': model_mean,
                'relative_performance': relative_performance,
                'sample_size': len(scores)
            }

        # Calculate metrics per model and problem level
        level_metrics = {}
        for level, models in level_scores.items():
            level_metrics[level] = {}
            for model, scores in models.items():
                if not scores:
                    # Still include entry with zeroes if we have best-of-N info
                    best_list = level_best_successes.get(level, {}).get(model, [])
                    level_metrics[level][model] = {
                        'mean_score': 0.0,
                        'best_of_n': (sum(best_list) / len(best_list)) if best_list else 0.0,
                        'sample_size': 0
                    }
                    continue

                model_mean = sum(scores) / len(scores)
                best_list = level_best_successes.get(level, {}).get(model, [])
                level_metrics[level][model] = {
                    'mean_score': model_mean,
                    'best_of_n': (sum(best_list) / len(best_list)) if best_list else 0.0,
                    'sample_size': len(scores)
                }

        return metrics, level_metrics

    def calculate_verification_metrics(self, problems):
        verification_stats = defaultdict(lambda: {'correct': 0, 'incorrect': 0, 'unknown': 0})

        for problem in problems:
            if 'model_solutions' not in problem or not problem['model_solutions']:
                continue
            for solution in problem['model_solutions']:
                for attempt in solution['attempts'][:self.attempts_included_per_model]:
                    verifier_results = self._get_verifier_results(attempt)
                    if not verifier_results:
                        continue

                    for result in verifier_results:
                        status = result.get('verified')
                        category = 'correct' if status == 1 else 'incorrect' if status == 0 else 'unknown'
                        verification_stats[solution['model']][category] += 1

        # Compute success rate
        for model, stats in verification_stats.items():
            total_verified = stats['correct'] + stats['incorrect']
            stats['success_rate'] = stats['correct'] / total_verified if total_verified > 0 else 0

        return dict(verification_stats)

    def calculate_best_of_n_metrics(self, problems, n: int) -> Dict[str, float]:
        """Compute best-of-N verification accuracy per model.

        For each model, treat each problem as an independent Bernoulli where success
        is 1 if any of the first N attempts for that problem verify correctly.
        Accuracy is averaged over problems that have at least one attempt.
        """
        model_to_problem_success: Dict[str, list] = defaultdict(list)

        for problem in problems:
            for solution in problem.get('model_solutions', []):
                model = solution.get('model')
                attempts = solution.get('attempts', [])
                if self.attempts_included_per_model is not None:
                    attempts = attempts[:self.attempts_included_per_model]

                # Consider up to N attempts for best-of-N
                considered = attempts[:n]
                if not considered:
                    continue

                any_correct = 0
                has_defined_verdict = False  # Align denominator with success_rate

                for attempt in considered:
                    verifier_results = self._get_verifier_results(attempt)
                    if not verifier_results:
                        continue

                    for vr in verifier_results:
                        verdict = vr.get('verified')
                        if verdict is None:
                            # Treat missing / unknown verdicts like success_rate: skip them
                            continue

                        has_defined_verdict = True
                        if verdict == 1:
                            any_correct = 1
                            break

                    if any_correct:
                        break

                # If there was no clear 0/1 verdict for this problem, skip it entirely
                # so that best-of-N uses the same effective set of problems as success_rate.
                if not has_defined_verdict:
                    continue

                model_to_problem_success[model].append(any_correct)

        results: Dict[str, float] = {}
        for model, successes in model_to_problem_success.items():
            if successes:
                results[model] = sum(successes) / len(successes)
            else:
                results[model] = 0.0
        return results