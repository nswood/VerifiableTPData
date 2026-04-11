from datetime import datetime, timezone
import time
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm
import simplejson as json  # for compatibility

import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
try:
    from vllm import LLM, SamplingParams
except ImportError:
    LLM = None
    SamplingParams = None

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import asyncio

from .problem_processor_base import BaseProblemProcessor

class SolutionGenerator(BaseProblemProcessor):
    """
    Generates solution attempts for problems using LLM models.
    
    Each generated attempt includes:
    - timestamp: ISO format timestamp of generation
    - prompt_type: Type of prompt used (e.g., "CoT", "standard")
    - detailed_solution: The generated solution text with mathematical reasoning and Python code
    
    Supports both API-based models (OpenAI, Gemini, Claude, Together AI) and local models via vLLM.
    Can generate multiple attempts in batch for efficiency.
    """
    def __init__(
            self,
            model_name: str,
            prompt_type: str = "CoT",
            config_path: str = "configs/inference_config.json",
            problems_dir: str = "data/tpbench",
            multi_gpu: int = 1,
            overwrite_attempts: bool = False,
            batch_size: Optional[int] = None,
            temperature: Optional[float] = None,
            requests_per_minute: int = 1000,
            max_concurrent_requests: int = 500,
            quiet: bool = False,
            gpu_memory_utilization: float = 0.95,
            model_alias: Optional[str] = None,
            reasoning_level: Optional[str] = None,
    ):
        super().__init__(
            model_name=model_name,
            config_path=config_path,
            problems_dir=problems_dir,
            multi_gpu=multi_gpu,
            quiet=quiet,
            temperature=temperature,
            gpu_memory_utilization=gpu_memory_utilization,
            model_alias=model_alias,
        )
        self.overwrite_attempts = overwrite_attempts
        self.prompt_type = prompt_type
        self.requests_per_minute = requests_per_minute
        self.max_concurrent_requests = max_concurrent_requests
        self.reasoning_level = reasoning_level
        
        if batch_size is None:
            if self._is_api_based_model():
                # Set a large batch size for API models to allow continuous streaming
                # The async worker handles rate limiting (RPM) and concurrency
                self.batch_size = 10000
                print(f"API model detected. Auto-setting batch_size to {self.batch_size} for continuous throughput.")
            else:
                # vLLM will perform dynamic batching; let it handle all prompts at once
                self.batch_size = None
                print("Local vLLM model detected. Using dynamic batching (no batch size limit).")
        else:
            self.batch_size = batch_size


        self.semaphore = threading.Semaphore(self.max_concurrent_requests)

    def _prepare_problem_generation(self, problem_path, num_attempts: int) -> Dict:
        """
        Load a problem and compute how many attempts still need to be generated.
        Returns a metadata dict used by both single-problem and global batching paths.
        """
        problem = self._load_problem(problem_path)
        if "model_solutions" not in problem:
            problem["model_solutions"] = []

        model_solution = next(
            (ms for ms in problem["model_solutions"] if ms.get("model") == self.model),
            None,
        )

        existing_attempts_count = 0
        if (not self.overwrite_attempts
                and model_solution
                and model_solution.get("attempts", [])):
            existing_attempts_count = len(model_solution.get("attempts", []))
            attempt_to_generate = max(0, num_attempts - existing_attempts_count)
        else:
            attempt_to_generate = num_attempts

        user_prompt = self._build_user_prompt(problem)

        return {
            "problem_path": problem_path,
            "problem": problem,
            "model_solution": model_solution,
            "existing_attempts_count": existing_attempts_count,
            "attempt_to_generate": attempt_to_generate,
            "user_prompt": user_prompt,
            "new_attempts": [],
        }

    def _build_user_prompt(self, problem):
        """
        Build the user prompt for solution generation.
        
        The prompt instructs the model to:
        1. First solve the problem mathematically
        2. Then convert the solution to Python code
        3. Follow specific code format requirements
        
        Args:
            problem: Problem dictionary containing problem_details with Problem Statement and Answer Requirements
            
        Returns:
            Formatted prompt string
        """
        combined_prompt = f"""Problem:
        {problem["problem_details"]["Problem Statement"]}

        IMPORTANT SOLUTION REQUIREMENTS:
        1. You MUST FIRST solve this problem using mathematical reasoning and symbolic calculations:
           - Use proper mathematical notation and symbols
           - Arrive at a final symbolic mathematical expression

        2. ONLY AFTER completing the mathematical solution:
           - Convert your final mathematical expression into Python code
           - The code must satisfy these requirements:
        {problem["problem_details"]["Answer Requirements"]}

        Code Format Requirements:
        1. Your solution MUST include the final executable Python code as required by the "Answer Requirements"
        2. You MUST wrap the final Python code between ```python and ``` tags
        3. Ensure the code is complete and can run independently
        4. The code should NOT contain ANY externally defined variables, including physical constants.
        5. The code MUST be concise and lightweight, faithfully and directly translating the final symbolic mathematical expression derived in step 1. Do NOT perform any further calculations or simplifications within the Python code itself.
        6. The code MUST NOT include any redundant elements such as conditional logic (if statements), exception handling (try/except blocks), or unnecessary checks.
        """
        return combined_prompt

    def _get_system_prompt(self) -> str:
        """
        Return the system prompt for this generator.
        Problem-type specific prompts have been removed; config now stores
        a single string per prompt_type (e.g., config["system_prompt"]["CoT"]).

        For oss models, prepends "Reasoning: <level>" if reasoning_level is set.
        """
        sp_cfg = self.config.get("system_prompt", {})
        val = sp_cfg.get(self.prompt_type)
        # New format: direct string
        if isinstance(val, str):
            base_prompt = val
        else:
            base_prompt = "System prompt not found"

        # Prepend reasoning level for oss models
        if self.reasoning_level and "oss" in self.model_name:
            return f"Reasoning: {self.reasoning_level}\n\n{base_prompt}"
        return base_prompt

    def generate_solution(self, problem_path, num_attempts: int = 1) -> List[Dict]:
        """
        Generate multiple solution attempts for a problem.
        
        Each attempt includes:
        - timestamp: ISO format timestamp
        - prompt_type: Type of prompt used (e.g., "CoT")
        - detailed_solution: The generated solution text
        
        Args:
            problem_path: Path to the problem JSON file
            num_attempts: Number of solution attempts to generate
            
        Returns:
            List of attempt dictionaries, each containing timestamp, prompt_type, and detailed_solution
        """
        info = self._prepare_problem_generation(problem_path, num_attempts)
        problem = info["problem"]
        user_prompt = info["user_prompt"]
        model_solution = info["model_solution"]
        existing_attempts_count = info["existing_attempts_count"]
        attempt_to_generate = info["attempt_to_generate"]
        
        if attempt_to_generate == 0:
            print(
                f"  ✗ Skipping generation for {problem_path} as {self.model} already has {existing_attempts_count} attempt(s) (required: {num_attempts})")
            return model_solution["attempts"] if model_solution else []

        solution_entries = []
        print(
            f"Attempting to generate {attempt_to_generate} solution(s)")
        
        # Determine if we can use batch generation
        can_use_batch = attempt_to_generate > 1
        if self.local_llm is not None:
            # Let vLLM handle all prompts in one dynamic batch
            effective_batch_size = attempt_to_generate
        else:
            effective_batch_size = max(1, self.batch_size or 1)
        
        # Try batch generation first when applicable
        batch_results: Optional[List[Optional[Dict]]] = None
        if can_use_batch:
            try:
                prompts = [user_prompt] * attempt_to_generate
                # If more attempts than batch_size, process in chunks
                if self._is_api_based_model():
                    print(f"Using API batch generation for {attempt_to_generate} attempt(s)")
                    chunked_results: List[Optional[Dict]] = []
                    for start in range(0, attempt_to_generate, effective_batch_size):
                        end = min(start + effective_batch_size, attempt_to_generate)
                        sub_prompts = prompts[start:end]
                        sub_results = self._generate_api_based_solutions_batch(sub_prompts)
                        chunked_results.extend(sub_results)
                    batch_results = chunked_results
                elif self.local_llm is not None:
                    print(f"Using vLLM dynamic batch generation for {attempt_to_generate} attempt(s)")
                    batch_results = self._generate_vllm_solutions_batch(prompts)
            except Exception as e:
                print(f"Batch generation failed: {e}")
                batch_results = None
        
        # If batch generation succeeded, create all solution entries from batch results
        if batch_results is not None and len(batch_results) > 0:
            for solution in batch_results:
                if solution is None:
                    continue
                
                solution_entry = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "prompt_type": self.prompt_type
                }
                solution_entry.update(solution)
                solution_entries.append(solution_entry)
        
        # Otherwise, use loop for single attempts or when batch failed
        else:
            for i in range(attempt_to_generate):
                solution = {}  # Initialize solution dict
                print(f"Generating attempt {i + 1}/{attempt_to_generate}...")

                try:
                    # Single generation (fallback when batch failed or attempt_to_generate == 1)
                    if self._is_api_based_model():
                        solution = self._generate_api_based_solution(user_prompt)
                        time.sleep(3)  # Rate limiting for API calls
                    else:
                        solution = self._generate_vllm_solution(user_prompt)
                except Exception as e:
                    print(f"Error during generation for attempt {i + 1}: {e}")
                    print(f"Error line: {e.__traceback__.tb_lineno}")
                    continue  # Skip this attempt if there's an error

                # Create solution entry dictionary
                solution_entry = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "prompt_type": self.prompt_type
                }
                
                # Merge generated solution content into the entry
                solution_entry.update(solution)

                solution_entries.append(solution_entry)

        # Update problem data structure
        if model_solution:
            if self.overwrite_attempts:
                model_solution["attempts"] = solution_entries
                print(f"Overwriting existing attempts for model {self.model}.")
            else:
                model_solution["attempts"].extend(solution_entries)
                print(f"Appending {len(solution_entries)} new attempts for model {self.model}.")
        else:
            problem["model_solutions"].append({
                "model": self.model,
                "attempts": solution_entries
            })
            print(f"Adding {len(solution_entries)} attempts for new model {self.model}.")

        self._save_problem(problem, problem_path)
        return solution_entries

    def _generate_api_based_solution(self, user_prompt_: str) -> Dict:
        # Build the common prompt/message list
        system_prompt = self._get_system_prompt()
        prompt_data = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt_},
        ]
        # Determine API call based on the model type
        if self._should_use_openai_api():
            content = self._generate_openai_response(prompt_data)
        elif self._should_use_claude_api():
            content = self._generate_claude_response(user_prompt_, system_prompt)
        elif self._should_use_grok_api():
            content = self._generate_grok_response(prompt_data)
        else:
            content = self._generate_gemini_response(user_prompt_)

        if not content:
            # Non-2xx HTTP responses (400/401/429/etc.) are surfaced by the SDKs
            # as exceptions, so reaching here corresponds to a successful HTTP 200.
            raise ValueError(f"Content is empty for model: {self.model_name}")
        return {
            "detailed_solution": content
        }

    def _generate_api_based_solutions_batch(self, user_prompts: List[str], pbar: Optional[tqdm] = None, on_progress: Optional[callable] = None) -> List[Dict]:
        """
        Refactored to use Asyncio for high-performance concurrency (500+ reqs).
        Uses explicit ThreadPoolExecutor to ensure performance across all environments (including Notebooks).
        """
        if not user_prompts:
            return []

        async def async_worker(index, prompt, semaphore, limiter_delay, executor, loop):
            async with semaphore:
                # Non-blocking Rate Limit Logic
                # (Simple implementation: just wait the delay * concurrency factor loosely)
                # For strict rate limiting, use library 'aiolimiter'
                await asyncio.sleep(limiter_delay)
                
                try:
                    # FIX: Use run_in_executor to FORCE usage of our large thread pool
                    # instead of asyncio.to_thread which uses the default executor (often limited)
                    result = await loop.run_in_executor(
                        executor, 
                        self._generate_api_based_solution, 
                        prompt
                    )
                except Exception as e:
                    print(f"Error: {e}")
                    result = None
                
                if pbar is not None:
                    pbar.update(1)
                
                if on_progress:
                    # Run callback immediately (note: this runs on the event loop)
                    try:
                        on_progress(index, result)
                    except Exception as e:
                        print(f"Error in on_progress callback: {e}")

                return result

        async def run_batch():
            # 1. Create Async Semaphore (Better than threading.Semaphore)
            sem = asyncio.Semaphore(self.max_concurrent_requests)
            
            # 2. Calculate delay for crude rate limiting
            # If 1000 RPM, delay is 0.06s per request.
            delay = 60.0 / max(1, self.requests_per_minute)
            
            # FIX: Define Executor Here explicitly
            with ThreadPoolExecutor(max_workers=self.max_concurrent_requests) as executor:
                loop = asyncio.get_running_loop()
                tasks = []
                for i, prompt in enumerate(user_prompts):
                    # Pass executor/loop to worker
                    task = asyncio.create_task(async_worker(i, prompt, sem, delay, executor, loop))
                    tasks.append(task)
                    
                    await asyncio.sleep(delay)
                
                return await asyncio.gather(*tasks)

        # Run the async loop inside your sync code
        print(f"🚀 Starting Async Batch for {len(user_prompts)} prompts...")
        
        try:
            # Try asyncio.run first (standard script usage)
            results = asyncio.run(run_batch())
        except RuntimeError:
            # Fallback for Notebooks/Existing Loops
            # We don't need to manually set the executor because we now pass our explicit 
            # executor directly to run_in_executor inside the worker.
            loop = asyncio.get_event_loop()
            results = loop.run_until_complete(run_batch())
        
        return results

    def _generate_vllm_solution(self, user_prompt_: str) -> Dict:
        messages = [
            {"role": "system", "content": self._get_system_prompt()},
            {"role": "user", "content": user_prompt_},
        ]

        # Check if tokenizer supports add_generation_prompt parameter
        # Mistral tokenizers don't support this parameter
        try:
            prompt_text = self.local_tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False
            )
        except (TypeError, Exception) as e:
            # Catch both TypeError and generic exceptions for Mistral tokenizers
            if "add_generation_prompt" in str(e):
                # Fallback for tokenizers that don't support add_generation_prompt
                prompt_text = self.local_tokenizer.apply_chat_template(
                    messages,
                    tokenize=False
                )
            else:
                raise
        sampling_kwargs = {"max_tokens": self.model_len}
        if self.temperature is not None:
            sampling_kwargs["temperature"] = self.temperature
        sampling_params = SamplingParams(**sampling_kwargs)

        output_sequence = self.local_llm.generate(
            prompt_text,
            sampling_params=sampling_params
        )

        output_text = output_sequence[0].outputs[0].text

        return {
            "detailed_solution": output_text
        }

    def _generate_vllm_solutions_batch(self, user_prompts: List[str]) -> List[Dict]:
        """Generate multiple solutions in a single vLLM batch call for speed.

        Args:
            user_prompts: A list of user prompts (one per attempt)

        Returns:
            A list of dicts, each containing 'detailed_solution'.
        """
        if not user_prompts:
            return []

        messages_list = [
            [
                {"role": "system", "content": self._get_system_prompt()},
                {"role": "user", "content": up},
            ]
            for up in user_prompts
        ]

        # Check if tokenizer supports add_generation_prompt parameter
        # Mistral tokenizers don't support this parameter
        prompt_texts: List[str] = []
        for messages in messages_list:
            try:
                prompt_text = self.local_tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=False,
                )
            except (TypeError, Exception) as e:
                # Catch both TypeError and generic exceptions for Mistral tokenizers
                if "add_generation_prompt" in str(e):
                    # Fallback for tokenizers that don't support add_generation_prompt
                    prompt_text = self.local_tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                    )
                else:
                    raise
            prompt_texts.append(prompt_text)

        sampling_kwargs = {"max_tokens": self.model_len}
        if self.temperature is not None:
            sampling_kwargs["temperature"] = self.temperature
        sampling_params = SamplingParams(**sampling_kwargs)

        outputs = self.local_llm.generate(
            prompt_texts,
            sampling_params=sampling_params,
        )

        results: List[Dict] = []
        for out in outputs:
            # Each out corresponds to one input prompt
            text = out.outputs[0].text if out.outputs else ""
            results.append(
                {
                    "detailed_solution": text
                }
            )

        return results


    def _print_generating(self):
        print(f"  Generating verifiable solutions using {self.model}...")

    def generate_solutions_for_problem(self,
                                       problem_path,
                                       num_attempts: int) -> None:
        """
        Generate solutions for a single problem.
        
        Args:
            problem_path: Path to the problem JSON file
            num_attempts: Number of solution attempts to generate
        """
        print(f"\nProcessing problem: {problem_path}")

        try:
            self._print_generating()
            solutions = self.generate_solution(
                problem_path,
                num_attempts=num_attempts
            )
            print(
                f"  ✓ Generated {len(solutions)} solutions for {self.model}")
        except Exception as e:
            print(f"  ✗ Error with {self.model}: {str(e)}")
            print(f"Error at line {e.__traceback__.tb_lineno}")

        print(f"Completed processing for {problem_path}")

    def _save_finished_problem(self, info: Dict):
        """Helper to save a problem when all its attempts are finished."""
        problem = info["problem"]
        model_solution = info["model_solution"]
        new_attempts = info["new_attempts"]
        problem_path = info["problem_path"]

        if not new_attempts:
            # No successful attempts for this problem
            print(f"  ✗ No successful attempts generated for {problem_path}")
            info["saved"] = True
            return

        if model_solution:
            if self.overwrite_attempts:
                model_solution["attempts"] = new_attempts
                print(f"Overwriting existing attempts for model {self.model} in {problem_path}.")
            else:
                model_solution.setdefault("attempts", [])
                model_solution["attempts"].extend(new_attempts)
                print(f"Appending {len(new_attempts)} new attempts for model {self.model} in {problem_path}.")
        else:
            problem.setdefault("model_solutions", [])
            problem["model_solutions"].append({
                "model": self.model,
                "attempts": new_attempts
            })
            print(f"Adding {len(new_attempts)} attempts for new model {self.model} in {problem_path}.")

        try:
            self._save_problem(problem, problem_path)
        except Exception as e:
            print(f"Error saving problem {problem_path}: {str(e)}")
            print("Skipping save error and continuing with others...")

        info["saved"] = True

    def generate_solutions_for_all_problems(self,
                                            num_attempts: int) -> None:
        """
        Generate solutions for all problems in the problems directory.
        
        Args:
            num_attempts: Number of solution attempts to generate per problem
        """
        problem_files = self.get_problem_files()

        # First pass: determine how many attempts are needed globally and
        # prepare metadata for each problem.
        problems_info: List[Dict] = []
        total_attempts = 0

        for problem_file in problem_files:
            problem_path = self.problems_dir / problem_file
            try:
                info = self._prepare_problem_generation(problem_path, num_attempts)
            except Exception as e:
                print(f"Error loading problem {problem_path}: {str(e)}")
                print("Skipping this problem and continuing with others...")
                continue

            attempt_to_generate = info["attempt_to_generate"]
            existing_attempts_count = info["existing_attempts_count"]

            if attempt_to_generate == 0:
                print(
                    f"  ✗ Skipping generation for {problem_path} as {self.model} already has "
                    f"{existing_attempts_count} attempt(s) (required: {num_attempts})"
                )
                continue

            # Track progress and save status per problem so we can flush each problem
            # to disk as soon as all its attempts are processed.
            info["attempts_processed"] = 0
            info["saved"] = False

            problems_info.append(info)
            total_attempts += attempt_to_generate

        if total_attempts == 0:
            print("No new attempts required for any problem. Exiting generation.")
            return

        print(f"Attempting to generate {total_attempts} solution(s) across {len(problems_info)} problem(s)")
        self._print_generating()

        # Build a flat list of "tasks" – one entry per attempt, referencing the
        # owning problem's info dict so we can map results back.
        tasks: List[Dict] = []
        for info in problems_info:
            for _ in range(info["attempt_to_generate"]):
                tasks.append(info)

        use_api = self._is_api_based_model()
        if not use_api and self.local_llm is not None:
            # Send all prompts at once to vLLM to leverage its dynamic batching
            effective_batch_size = len(tasks) if tasks else 1
        else:
            effective_batch_size = max(1, self.batch_size or 1)

        # Global tqdm over total attempts
        with tqdm(total=total_attempts, desc="Generating attempts") as pbar:
            idx = 0
            while idx < len(tasks):
                batch_infos = tasks[idx: idx + effective_batch_size]
                user_prompts = [bi["user_prompt"] for bi in batch_infos]

                batch_results: Optional[List[Optional[Dict]]] = None

                # Try batch generation first when beneficial
                if len(user_prompts) > 1:
                    try:
                        if use_api:
                            # Define callback for immediate processing
                            def batch_callback(idx, res):
                                info = batch_infos[idx]
                                
                                # FIX: ALWAYS increment processed count, even on failure
                                info["attempts_processed"] = info.get("attempts_processed", 0) + 1

                                if res is not None and res.get("detailed_solution"):
                                    solution_entry = {
                                        "timestamp": datetime.now(timezone.utc).isoformat(),
                                        "prompt_type": self.prompt_type,
                                    }
                                    solution_entry.update(res)
                                    info["new_attempts"].append(solution_entry)
                                
                                # Check saving immediately
                                attempt_to_generate = info.get("attempt_to_generate", 0)
                                attempts_processed = info.get("attempts_processed", 0)
                                if not info.get("saved") and attempts_processed >= attempt_to_generate:
                                    self._save_finished_problem(info)

                            batch_results = self._generate_api_based_solutions_batch(
                                user_prompts, 
                                pbar=pbar,
                                on_progress=batch_callback
                            )
                        elif self.local_llm is not None:
                            batch_results = self._generate_vllm_solutions_batch(user_prompts)
                            # Process results immediately for vLLM (similar to API callback)
                            for i, res in enumerate(batch_results):
                                info = batch_infos[i]

                                # Always increment processed count
                                info["attempts_processed"] = info.get("attempts_processed", 0) + 1

                                if res is not None and res.get("detailed_solution"):
                                    solution_entry = {
                                        "timestamp": datetime.now(timezone.utc).isoformat(),
                                        "prompt_type": self.prompt_type,
                                    }
                                    solution_entry.update(res)
                                    info["new_attempts"].append(solution_entry)

                                pbar.update(1)

                                # Save immediately when problem is complete
                                attempt_to_generate = info.get("attempt_to_generate", 0)
                                attempts_processed = info.get("attempts_processed", 0)
                                if not info.get("saved") and attempts_processed >= attempt_to_generate:
                                    self._save_finished_problem(info)
                    except Exception as e:
                        print(f"Batch generation failed for attempts {idx + 1}-{idx + len(user_prompts)}: {e}")
                        batch_results = None

                # Fallback to per-attempt generation if batch failed or only one prompt
                if batch_results is None:
                    batch_results = []
                    for prompt, info in zip(user_prompts, batch_infos):
                        try:
                            if use_api:
                                result = self._generate_api_based_solution(prompt)
                                # Basic rate limiting between single API calls
                                time.sleep(3)
                            else:
                                result = self._generate_vllm_solution(prompt)
                        except Exception as e:
                            print(f"Error during generation for {info['problem_path']}: {e}")
                            result = None
                        batch_results.append(result)

                # Post-processing (Only needed for fallback cases)
                # Skip if API batch succeeded (callback handled) or vLLM batch succeeded (loop above handled)
                batch_succeeded = (len(user_prompts) > 1 and batch_results is not None)

                if not batch_succeeded:
                    # Attach results back to their problems and update per-problem progress
                    for info, solution in zip(batch_infos, batch_results):
                        # Count this attempt as processed regardless of success
                        info["attempts_processed"] = info.get("attempts_processed", 0) + 1

                        if solution is not None and solution.get("detailed_solution"):
                            solution_entry = {
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                                "prompt_type": self.prompt_type,
                            }
                            solution_entry.update(solution)
                            info["new_attempts"].append(solution_entry)
                        
                        # If not using API batch (which updates pbar internally), update here
                        if not (use_api and len(user_prompts) > 1):
                            pbar.update(1)

                        # Check if this specific problem is now fully processed
                        attempt_to_generate = info.get("attempt_to_generate", 0)
                        attempts_processed = info.get("attempts_processed", 0)

                        if not info.get("saved") and attempts_processed >= attempt_to_generate:
                            self._save_finished_problem(info)

                idx += len(batch_infos)