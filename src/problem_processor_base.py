from datetime import datetime, timezone
import time
import signal
from contextlib import contextmanager

from pathlib import Path
from typing import Dict, List, Optional, Callable, Tuple, Any
from sympy import Symbol, Function, FunctionClass, sympify, Mul
import random
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
import numpy as np
import math
import ast
import inspect
import copy
import multiprocessing
from queue import Empty

import os
import tempfile
import fcntl


def load_problem_with_lock(problem_path: Path) -> Dict:
    """
    Load a problem JSON file using a shared file lock to avoid concurrent
    read/write conflicts.
    """
    try:
        lock_path = problem_path.with_suffix(problem_path.suffix + ".lock")
        lock_path.touch(exist_ok=True)
        with open(lock_path, "r+") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_SH)
            try:
                with open(problem_path, 'r', encoding="utf-8") as f:
                    return json.load(f)
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    except json.JSONDecodeError as e:
        # Read file content and show context around the error location
        with open(problem_path, 'r', encoding="utf-8") as f:
            content = f.readlines()

        # Get lines around the error line
        start_line = max(0, e.lineno - 3)
        end_line = min(len(content), e.lineno + 2)

        context = ''.join(f"{i+1}: {line}" for i, line in enumerate(content[start_line:end_line]))

        error_msg = (
            f"\nJSON decode error in file {problem_path}:\n"
            f"Error: {str(e)}\n"
            f"Around line {e.lineno}, column {e.colno}:\n"
            f"{context}\n"
            f"{'~' * (e.colno-1)}^\n"
        )
        raise json.JSONDecodeError(error_msg, e.doc, e.pos) from None
    except Exception as e:
        raise Exception(f"Error loading problem from {problem_path}: {str(e)}") from None


def save_problem_with_lock(problem: Dict, problem_path: Path) -> None:
    """
    Save a problem dictionary to a JSON file with file locking and intelligent merging.

    This function:
    - Uses file locking to prevent concurrent write conflicts
    - Merges with existing file content to avoid data loss
    - Cleans NaN/Inf values before saving
    - Uses atomic file replacement for safety
    """
    def clean_nan(obj):
        """Recursively clean NaN/Inf values in data, supporting more data types."""
        try:

            if isinstance(obj, dict):
                return {k: clean_nan(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [clean_nan(item) for item in obj]
            elif isinstance(obj, np.ndarray):
                obj = obj.astype(float, copy=True)
                obj[~np.isfinite(obj)] = None
                return obj.tolist()
            elif isinstance(obj, (float, np.floating)):
                if np.isnan(obj) or np.isinf(obj):
                    return None
                return float(obj)
            elif isinstance(obj, (int, np.integer)):
                return int(obj)
            elif isinstance(obj, str):
                lower_str = obj.lower()
                if lower_str in ('inf', '-inf', 'nan', 'infinity', '-infinity'):
                    return None
                return obj
            return obj
        except Exception as e:
            print(f"Warning: Error cleaning value {obj}: {str(e)}")
            return None

    def _merge_model_solutions(base_ms: List[Dict], incoming_ms: List[Dict]) -> List[Dict]:
        """
        Merge model solutions from disk (base) and memory (incoming).

        Uses content-based matching to identify attempts:
        - Matches by hash of detailed_solution[:500] + timestamp
        - Updates existing attempts instead of creating duplicates
        - Handles race conditions in multiprocessing scenarios
        """
        model_to_base = {ms.get("model"): copy.deepcopy(ms) for ms in (base_ms or [])}
        for inc in (incoming_ms or []):
            model_name = inc.get("model")
            if model_name in model_to_base:
                base_entry = model_to_base[model_name]
                base_attempts = list(base_entry.get("attempts", []))
                incoming_attempts = inc.get("attempts", []) or []

                def get_attempt_key(attempt: dict) -> tuple:
                    """Generate a unique key for an attempt based on its content."""
                    if not isinstance(attempt, dict):
                        return None
                    detailed_solution = attempt.get("detailed_solution", "")
                    timestamp = attempt.get("timestamp")
                    solution_hash = hash(detailed_solution[:500]) if detailed_solution else None
                    return (solution_hash, timestamp) if solution_hash else (timestamp,) if timestamp else None

                base_attempt_map = {}
                for idx, base_attempt in enumerate(base_attempts):
                    if isinstance(base_attempt, dict):
                        key = get_attempt_key(base_attempt)
                        if key:
                            if key not in base_attempt_map:
                                base_attempt_map[key] = []
                            base_attempt_map[key].append(idx)

                matched_base_indices = set()

                for incoming_attempt in incoming_attempts:
                    if not isinstance(incoming_attempt, dict):
                        continue

                    key = get_attempt_key(incoming_attempt)

                    if key and key in base_attempt_map:
                        matched = False
                        for base_idx in base_attempt_map[key]:
                            if base_idx not in matched_base_indices:
                                base_attempts[base_idx] = {**base_attempts[base_idx], **incoming_attempt}
                                matched_base_indices.add(base_idx)
                                matched = True
                                break

                        if not matched:
                            base_attempts.append(incoming_attempt)
                    else:
                        base_attempts.append(incoming_attempt)

                merged_entry = {**base_entry, **inc}
                merged_entry["attempts"] = base_attempts
                model_to_base[model_name] = merged_entry
            else:
                model_to_base[model_name] = copy.deepcopy(inc)
        return list(model_to_base.values())

    try:
        lock_path = problem_path.with_suffix(problem_path.suffix + ".lock")
        lock_path.touch(exist_ok=True)
        with open(lock_path, "r+") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                current_on_disk = {}
                if problem_path.exists():
                    with open(problem_path, 'r', encoding='utf-8') as cur_f:
                        current_on_disk = json.load(cur_f)

                cleaned_incoming = clean_nan(problem)

                if isinstance(current_on_disk, dict) and isinstance(cleaned_incoming, dict):
                    merged = copy.deepcopy(current_on_disk)
                    for k, v in cleaned_incoming.items():
                        if k == "model_solutions" and isinstance(v, list):
                            merged[k] = _merge_model_solutions(current_on_disk.get(k, []), v)
                        else:
                            merged[k] = v
                else:
                    merged = cleaned_incoming

                problem_path.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(
                    'w',
                    delete=False,
                    dir=str(problem_path.parent),
                    prefix=problem_path.name + '.',
                    suffix='.tmp',
                    encoding='utf-8'
                ) as tmp_f:
                    json.dump(merged, tmp_f, indent=4, ensure_ascii=False, ignore_nan=True)
                    tmp_f.flush()
                    os.fsync(tmp_f.fileno())
                    temp_name = tmp_f.name
                os.replace(temp_name, problem_path)
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    except Exception as e:
        print(f"Error saving problem to {problem_path}: {str(e)}")
        print("Problem data structure:")
        print(json.dumps(problem, indent=2, default=lambda x: f"<{type(x).__name__}>"))
        raise


class BaseProblemProcessor:
    """
    Base class for processing problems (generating solutions or grading).

    Handles:
    - Loading and saving problem files with file locking
    - Initializing API clients or local LLM models
    - Merging attempts intelligently to avoid duplicates
    - Managing subprocesses for parallel processing
    """
    def __init__(
            self,
            model_name: str,
            config_path: str,
            problems_dir: str = "data/tpbench",
            multi_gpu: int = 1,
            quiet: bool = False,
            temperature: Optional[float] = None,
            gpu_memory_utilization: float = 0.95,
            model_alias: Optional[str] = None,
            api_base_url: Optional[str] = None,
    ):
        self.project_root = Path(__file__).parent.parent
        self.problems_dir = self.project_root / problems_dir
        self.config = self._load_config(self.project_root / config_path)

        # model_name is used for loading the model
        # model (or model_alias) is used for display/saving in results
        self.model_name = model_name.split("@")[0]
        self.model = model_alias if model_alias else model_name
        self.method = model_name.split("@")[1] if "@" in model_name else None

        self.multi_gpu = multi_gpu
        self.model_len = 32768
        self.subprocesses = []
        self.quiet = quiet
        self.temperature = temperature
        self.gpu_memory_utilization = gpu_memory_utilization
        self.api_base_url = api_base_url  # Custom API base URL (e.g., vLLM server)
        self.local_llm = None  # Initialize to None, will be set by _load_vllm_model() for local models
        self.local_tokenizer = None  # Initialize to None, will be set by _load_vllm_model() for local models
        self._init_client()

    def _load_config(self, config_path: Path) -> Dict:
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found at: {config_path}")
        with open(config_path, 'r', encoding="utf-8") as f:
            return json.load(f)

    def _init_client(self):
        api_keys_path = self.project_root / "api_keys/api_keys.json"
        if api_keys_path.exists():
            with open(api_keys_path, 'r', encoding="utf-8") as f:
                keys = json.load(f)
                self.openai_api_key = keys.get("openai_api_key", None)
                self.together_api_key = keys.get("together_api_key", None)
                self.gemini_api_key = keys.get("gemini_api_key", None)
                self.anthropic_api_key = keys.get("anthropic_api_key", None)
                self.grok_api_key = keys.get("grok_api_key", None)

        # Custom API base URL (e.g., vLLM server) — use OpenAI client directly
        if self.api_base_url:
            if not self.quiet:
                print(f"Using custom API base: {self.api_base_url} for {self.model_name}")
            try:
                from openai import OpenAI
            except ImportError as e:
                raise ImportError("The 'openai' package is required but not installed. Please install it with 'pip install openai'.") from e
            self.api_client = OpenAI(base_url=self.api_base_url, api_key="EMPTY")
            return

        if self._is_api_based_model():
            if self.use_together_api():
                if not self.quiet:
                    print(f"Using together API for {self.model_name}")
                try:
                    from together import Together
                except ImportError as e:
                    raise ImportError("The 'together' package is required but not installed. Please install it with 'pip install together'.") from e
                self.api_client = Together(api_key=self.together_api_key)
            elif self.model_name.startswith("claude"):
                if not self.quiet:
                    print(f"Using anthropic API for {self.model_name}")
                try:
                    import anthropic
                except ImportError as e:
                    raise ImportError("The 'anthropic' package is required but not installed. Please install it with 'pip install anthropic'.") from e
                self.api_client = anthropic.Anthropic(api_key=self.anthropic_api_key)
            elif self.model_name.startswith("gemini"):
                if not self.quiet:
                    print(f"Using gemini API for {self.model_name}")
                try:
                    from google import genai
                except ImportError as e:
                    raise ImportError("The 'google-genai' SDK is required but not installed. Please install it with 'pip install google-genai'.") from e
                self.api_client = genai.Client(api_key=self.gemini_api_key)
            elif self.model_name.startswith("grok"):
                if not self.quiet:
                    print(f"Using xAI (Grok) API for {self.model_name}")
                try:
                    from xai_sdk import Client
                except ImportError as e:
                    raise ImportError("The 'xai-sdk' package is required but not installed. Please install it with 'pip install xai-sdk'.") from e
                # Grok uses the native xAI SDK
                self.api_client = Client(api_key=self.grok_api_key, timeout=3600)  # 1 hour timeout for reasoning models
            else:
                if not self.quiet:
                    print(f"Using openai API for {self.model_name}")
                try:
                    from openai import OpenAI
                except ImportError as e:
                    raise ImportError("The 'openai' package is required but not installed. Please install it with 'pip install openai'.") from e
                self.api_client = OpenAI(api_key=self.openai_api_key)
        else:
            self._load_vllm_model()

    def _is_api_based_model(self) -> bool:
        if self.api_base_url:
            return True
        if self.model_name.startswith("deepseek-ai/DeepSeek-R1-Distill"):
            return False
        if self.model_name == "openai/gpt-oss-120b" or self.model_name == "openai/gpt-oss-20b":
            return True
        # Check if model is an API-based model (not a local model)
        return self.model_name.startswith(("chatgpt", "gpt-", "o1", "o3", "o4", "deepseek", "claude", "gemini", "moonshotai", "grok"))
    
    def _is_openai_model(self):
        if self.model_name == "gpt-5" or self.model_name.startswith(("o1", "o3", "o4")):   
            return "reasoning"
        elif self.model_name.startswith("gpt") or self.model_name.startswith("chatgpt"):
            return "classic"
        else:
            return False
    
    def use_together_api(self) -> bool:
        return self.model_name.startswith("deepseek") or self.model_name.startswith("moonshotai") or self.model_name == ("openai/gpt-oss-120b") #or self.model_name == ("openai/gpt-oss-20b")

    def _should_use_openai_api(self, model_name: Optional[str] = None) -> bool:
        """
        Check if the model should use OpenAI API (including Together AI).

        Args:
            model_name: Optional model name (defaults to self.model_name)

        Returns:
            True if the model should use OpenAI API, False otherwise
        """
        if self.api_base_url:
            return True
        if model_name is None:
            model_name = self.model_name
        return (self._is_openai_model() != False or
                model_name.startswith("deepseek") or
                model_name.startswith("moonshotai") or
                model_name == "openai/gpt-oss-120b" or
                model_name == "openai/gpt-oss-20b")
    
    def _should_use_claude_api(self, model_name: Optional[str] = None) -> bool:
        if model_name is None:
            model_name = self.model_name
        return model_name.startswith("claude")
    def _generate_openai_response(self, messages: List[Dict[str, str]], model_name: Optional[str] = None, api_client: Optional[Any] = None, temperature: Optional[float] = None, max_tokens: Optional[int] = None, response_format: Optional[Dict] = None) -> str:
        """
        Generate a response from OpenAI API (including Together AI).
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            model_name: Optional model name (defaults to self.model_name)
            api_client: Optional API client (defaults to self.api_client)
            temperature: Optional temperature (defaults to self.temperature)
            max_tokens: Optional max tokens (defaults to None, or 32768 for deepseek/together)
            response_format: Optional response format (e.g., {"type": "json_object"})
        
        Returns:
            The response text from OpenAI
        """
        if model_name is None:
            model_name = self.model_name
        if api_client is None:
            api_client = self.api_client
        if temperature is None:
            temperature = self.temperature
        
        # Determine model type based on model_name
        # Custom API base (e.g., vLLM) always uses classic mode
        if self.api_base_url:
            openai_model_type = "classic"
        elif model_name == "gpt-5" or model_name.startswith(("o1", "o3", "o4")) or "oss" in model_name:
            openai_model_type = "reasoning"
        else:
            openai_model_type = "classic"  # Default for together API models
        
        if openai_model_type == "reasoning":
            # OpenAI reasoning models (o1, o3, o4) - don't use temperature parameter
            kwargs = {
                "model": model_name,
                "messages": messages,
                "reasoning_effort": "medium",
            }
            if response_format:
                kwargs["response_format"] = response_format
            response = api_client.chat.completions.create(**kwargs)
            return response.choices[0].message.content
        else:
            # OpenAI classic models or Together AI - only include temperature if not None
            kwargs = {
                "model": model_name,
                "messages": messages,
            }
            if temperature is not None:
                kwargs["temperature"] = temperature
            # Set max_tokens for deepseek/together models if not provided
            if max_tokens is None and (model_name.startswith("deepseek") or
                                      model_name.startswith("moonshotai") or
                                      model_name == "openai/gpt-oss-120b" or
                                      model_name == "openai/gpt-oss-20b"):
                kwargs["max_tokens"] = 32768
            elif max_tokens is not None:
                kwargs["max_tokens"] = max_tokens
            if response_format:
                kwargs["response_format"] = response_format
            
            response = api_client.chat.completions.create(**kwargs)
            return response.choices[0].message.content
    
    def _should_use_grok_api(self, model_name: Optional[str] = None) -> bool:
        if model_name is None:
            model_name = self.model_name
        return model_name.startswith("grok")

    def _generate_grok_response(self, messages: List[Dict[str, str]], model_name: Optional[str] = None, api_client: Optional[Any] = None) -> str:
        """
        Generate a response from Grok API using the native xAI SDK.

        Args:
            messages: List of message dicts with 'role' and 'content'
            model_name: Optional model name (defaults to self.model_name)
            api_client: Optional API client (defaults to self.api_client)

        Returns:
            The response text from Grok
        """
        if model_name is None:
            model_name = self.model_name
        if api_client is None:
            api_client = self.api_client

        # Import here to avoid circular imports
        from xai_sdk.chat import user, system

        # Create a chat session
        chat = api_client.chat.create(model=model_name)

        # Add messages to the chat
        for message in messages:
            if message["role"] == "system":
                chat.append(system(message["content"]))
            elif message["role"] == "user":
                chat.append(user(message["content"]))
            # Note: xAI SDK doesn't have assistant role appending in the same way

        # Sample the response
        response = chat.sample()
        return response.content

    def _generate_claude_response(self, user_content: str, system_prompt: str, model_name: Optional[str] = None, api_client: Optional[Any] = None, max_tokens: Optional[int] = None) -> str:
        """
        Generate a response from Claude API.

        Args:
            user_content: The user message content
            system_prompt: The system prompt
            model_name: Optional model name (defaults to self.model_name)
            api_client: Optional API client (defaults to self.api_client)
            max_tokens: Maximum tokens. Defaults to 128000 for Opus models, 32768 for others.

        Returns:
            The response text from Claude
        """
        if model_name is None:
            model_name = self.model_name
        if api_client is None:
            api_client = self.api_client

        # Use max effort + streaming for Claude Opus (required for long requests)
        is_opus = "opus" in model_name.lower()

        if max_tokens is None:
            max_tokens = 128000 if is_opus else 32768

        kwargs = {
            "model": model_name,
            "system": system_prompt,
            "messages": [
                {"role": "user", "content": user_content}
            ],
            "max_tokens": max_tokens,
        }
        if is_opus:
            kwargs["output_config"] = {"effort": "max"}

        if is_opus:
            # Streaming required for requests that may take >10 minutes
            with api_client.messages.stream(**kwargs) as stream:
                response = stream.get_final_message()
        else:
            response = api_client.messages.create(**kwargs)
        return response.content[0].text
    
    def _generate_gemini_response(self, contents: str, model_name: Optional[str] = None, api_client: Optional[Any] = None) -> str:
        """
        Generate a response from Gemini API with automatic Gemini 3 Pro configuration.
        """
        if model_name is None:
            model_name = self.model_name
        if api_client is None:
            api_client = self.api_client
        
        # Check if Gemini 3 Pro
        is_gemini_3_pro = (model_name.lower() == "gemini-3-pro-preview" or
                          model_name.lower().startswith("gemini-3-pro"))

        # 1. Define the base config dictionary
        # Note: In the new SDK, this argument is usually called 'config'
        generation_config = {}

        if is_gemini_3_pro and model_name.lower() != "gemini-3-pro-preview":
            # 2. Nest specific parameters inside the config dict
            # Skip thinking_config for gemini-3-pro-preview as it doesn't support it
            generation_config = {
                "temperature": 1.0,
                "thinking_config": {
                    "thinking_level": "high"
                }
            }
        response = api_client.models.generate_content(
            model=model_name,
            contents=contents,
            config=generation_config
        )
        return response.text

    def _load_vllm_model(self):
        if not self.quiet:
            print(f"Loading model: {self.model_name}")
            if self.model != self.model_name:
                print(f"Results will be saved as: {self.model}")

        # Ensure HuggingFace cache is set correctly (use environment variable or default)
        hf_cache = os.environ.get('HF_HOME') or os.environ.get('HF_HUB_CACHE') or os.environ.get('TRANSFORMERS_CACHE')
        if hf_cache:
            if not self.quiet:
                print(f"Using HuggingFace cache: {hf_cache}")
            # Set cache directory for transformers
            os.environ['HF_HOME'] = hf_cache
            os.environ['HF_HUB_CACHE'] = hf_cache
            os.environ['TRANSFORMERS_CACHE'] = hf_cache

        self.local_llm = LLM(
            model=self.model_name,
            tensor_parallel_size=self.multi_gpu,
            max_model_len=self.model_len,
            trust_remote_code=True,
            enforce_eager=True,
            # enforce_eager=False,
            gpu_memory_utilization=self.gpu_memory_utilization,
            quantization="AWQ" if "AWQ" in self.model_name else None,
        )
        self.local_tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        # Define default sampling params after model loading
        # Only include temperature if explicitly set (not None), otherwise use model default
        sampling_kwargs = {"max_tokens": self.model_len}
        if self.temperature is not None:
            sampling_kwargs["temperature"] = self.temperature
        self.default_sampling_params = SamplingParams(**sampling_kwargs)

    def _load_problem(self, problem_path: Path) -> Dict:
        """Instance wrapper around the shared lock-based loader."""
        return load_problem_with_lock(problem_path)

    def _save_problem(self, problem: Dict, problem_path: Path):
        """Instance wrapper around the shared lock-based saver."""
        save_problem_with_lock(problem, problem_path)

    def get_problem_files(self):
        files = list(self.problems_dir.glob("*.json"))
        problem_files = [file.relative_to(self.problems_dir) for file in files]
        if len(problem_files) == 0:
            print(f"No problem files found in {self.problems_dir}, exiting...")
            exit(0)
        return problem_files

    def _terminate_subprocesses(self):
        for proc in self.subprocesses:
            if proc.poll() is None:
                proc.terminate()
                proc.wait()
        self.subprocesses.clear()





