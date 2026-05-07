import argparse
import logging
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.latex_generator import LaTeXGenerator


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    parser = argparse.ArgumentParser(description="Generate LaTeX documents for physics problems.")

    parser.add_argument("--problems_dir", type=str, default=None, help="Directory containing problem JSONs (repo-relative). Required unless --problem_path is given.")
    parser.add_argument("--problem_path", type=str, help="Specific problem to generate LaTeX for. Optional.")

    parser.add_argument("--analysis_only", action="store_true", default=False, help="Flag to only perform analysis without generating a LaTeX document.")
    parser.add_argument("--visualize_problem_only", action="store_true", default=False, help="Flag to visualize the problem only, without generating full analysis.")

    parser.add_argument("--attempts_included_per_model", type=int, help="Number of attempts to include per model. Optional.")
    parser.add_argument("--num_attempts", type=int, help="Number of attempts to consider for rollout calculations (alias for --attempts_included_per_model). Optional.")
    parser.add_argument("--best_of_n", type=int, default=5, help="Best-of-N metric to report (capped by num_attempts). Default: %(default)s")
    parser.add_argument("--baseline_model", type=str, default="chatgpt-4o-latest", help="Baseline model to use for verification-based metrics. Default: %(default)s")
    parser.add_argument("--selected_solvers", type=str, nargs="+", help="List of selected solvers to include in the report. Optional.")

    # Topic filtering arguments
    parser.add_argument("--include_topics", type=str, nargs="+", help="Only include problems with these Topic Entry IDs (e.g., GR-01 GR-02). Optional.")
    parser.add_argument("--exclude_topics", type=str, nargs="+", help="Exclude problems with these Topic Entry IDs (e.g., GR-01 GR-02 GR-03). Optional.")

    parser.add_argument("--output_name", type=str, help="Name of the output file (without extension). Optional.")

    # PDF compilation flag
    parser.add_argument(
        "-pdf",
        "--pdf",
        dest="pdf",
        action="store_true",
        help="Compile the generated LaTeX file to PDF and leave the PDF next to the .tex file "
             "(removes auxiliary files like .aux, .log, .out, .toc).",
    )

    # Verification / visualization flags
    parser.add_argument("--show_verification", action="store_true", default=True, help="Show auto-verification results in the output. Default: %(default)s")
    parser.add_argument("--show_cot", action="store_true", default=False, help="Show DeepSeek-R1 CoT `<think>` content in the output. Default: %(default)s")
    parser.add_argument("--correct_only", action="store_true", default=False, help="Only include attempts that are verified as correct. Default: %(default)s")
    parser.add_argument("--show_model_performance_by_problem", action="store_true", default=False, help="Show the Model Performance by Problem table. Default: %(default)s")
    args = parser.parse_args()

    if args.problems_dir is None and args.problem_path is None:
        parser.error("Either --problems_dir or --problem_path must be provided.")

    # When only --problem_path is given, derive --problems_dir from its parent
    # so LaTeXGenerator's existence check has a concrete dir to look at.
    if args.problems_dir is None:
        problem_path = Path(args.problem_path).resolve()
        parent = problem_path.parent
        try:
            args.problems_dir = str(parent.relative_to(REPO_ROOT))
        except ValueError:
            args.problems_dir = str(parent)

    # Handle num_attempts as an alias for attempts_included_per_model
    # If both are specified, num_attempts takes precedence
    attempts_to_use = args.num_attempts if args.num_attempts is not None else args.attempts_included_per_model

    # Create an instance of LaTeXGenerator with the parsed arguments
    generator = LaTeXGenerator(
        problems_dir=args.problems_dir,
        problem_path=args.problem_path,
        visualize_problem_only=args.visualize_problem_only,
        analysis_only=args.analysis_only,
        baseline_model=args.baseline_model,
        attempts_included_per_model=attempts_to_use,
        best_of_n=args.best_of_n,
        selected_solvers=args.selected_solvers,
        output_name=args.output_name,
        show_verification=args.show_verification,
        show_cot_r1=args.show_cot,
        only_compile_correct_attempts=args.correct_only,
        show_model_performance_by_problem=args.show_model_performance_by_problem,
        include_topics=args.include_topics,
        exclude_topics=args.exclude_topics,
    )

    # Optional: log generator configuration at debug level
    logging.getLogger(__name__).debug(generator.__dict__)

    tex_path = generator.generate_latex()

    # Optionally compile to PDF
    if args.pdf and tex_path is not None:
        project_root = Path(__file__).resolve().parents[1]
        tex_path = Path(tex_path)

        if not tex_path.exists():
            logging.error("Tex file not found for PDF compilation: %s", tex_path)
            return

        output_dir = tex_path.parent
        name = tex_path.name

        logging.info("Compiling PDF for %s", tex_path)
        outputs = []
        rc = 0
        for _ in range(2):
            cmd = [
                "pdflatex",
                "-interaction=nonstopmode",
                "-file-line-error",
                "-output-directory",
                str(output_dir),
                str(tex_path),
            ]
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                cwd=str(project_root),
            )
            outputs.append(proc.stdout)
            rc = proc.returncode

        pdf_path = tex_path.with_suffix(".pdf")
        if not pdf_path.exists():
            logging.error("PDF compilation failed for %s. pdflatex output:\n%s", tex_path, "\n".join(outputs))
        else:
            logging.info("PDF created at %s", pdf_path)

        # Clean up auxiliary files (.aux, .log, .out, .toc, .pre) next to the tex file
        aux_exts = [".aux", ".log", ".out", ".toc", ".pre"]
        for ext in aux_exts:
            aux_file = tex_path.with_suffix(ext)
            if aux_file.exists():
                try:
                    aux_file.unlink()
                except Exception as e:
                    logging.warning("Failed to remove auxiliary file %s: %s", aux_file, e)


if __name__ == "__main__":
    main()
