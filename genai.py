import json
import os
import time
from typing import Iterable, List, Optional

from google import genai
from google.genai import types

# OpenAI support
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

API_KEY_FILE = "api_keys/api_keys.json"
API_KEY_NAME = "gemini_api_key"
GEMINI_MODEL_NAME = "gemini-2.5-pro"  # Ensure this is up-to-date


def read_api_key(key_name=API_KEY_NAME, config_file=API_KEY_FILE):
    """Reads the specified API key from the config file."""
    if not os.path.exists(config_file):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_file_alt = os.path.join(script_dir, config_file)
        if not os.path.exists(config_file_alt):
            config_file_alt_2 = os.path.join(script_dir, '..', '..', config_file)  # Go up two levels if needed
            if not os.path.exists(config_file_alt_2):
                raise FileNotFoundError(f"API key file not found: {config_file} or {config_file_alt} or {config_file_alt_2}")
            else:
                config_file = config_file_alt_2
        else:
            config_file = config_file_alt
            
    try:
        with open(config_file, 'r') as f:
            keys = json.load(f)
        api_key = keys.get(key_name)
        if not api_key:
            raise ValueError(f"'{key_name}' not found in {config_file}")
        return api_key
    except json.JSONDecodeError:
        raise ValueError(f"Error decoding JSON from {config_file}")
    except Exception as e:
        raise RuntimeError(f"An error occurred while reading the API key: {e}")


def call_gen_ai(prompt, api_key, model=None, system_prompt=None):
    """
    Calls the GenAI API (Gemini or OpenAI) with the given prompt.

    Args:
        prompt: The user prompt
        api_key: API key for the service
        model: Model name (defaults to GEMINI_MODEL_NAME for Gemini models)
        system_prompt: Optional system prompt (used for OpenAI models)

    Returns:
        The generated text response
    """
    model_name = model or GEMINI_MODEL_NAME

    # Check if this is an OpenAI model
    is_openai_model = (
        model_name.startswith("gpt") or
        model_name.startswith("o1") or
        model_name.startswith("o3") or
        model_name.startswith("chatgpt")
    )

    if is_openai_model:
        if OpenAI is None:
            raise ImportError("OpenAI package not installed. Install with: pip install openai")

        # Use OpenAI API
        client = OpenAI(api_key=api_key)

        # Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Determine if this is a reasoning model (o1, o3, o4)
        is_reasoning_model = model_name.startswith(("o1", "o3", "o4"))

        if is_reasoning_model:
            # Reasoning models don't use temperature
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                reasoning_effort="high"
            )
        else:
            # Classic OpenAI models
            response = client.chat.completions.create(
                model=model_name,
                messages=messages
            )

        return response.choices[0].message.content

    # Gemini API via google.genai SDK
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model_name,
        contents=prompt
    )
    if hasattr(response, 'text') and response.text:
        return response.text
    # Fallback for structured content
    if hasattr(response, 'candidates') and response.candidates:
        for part in response.candidates[0].content.parts:
            if hasattr(part, 'text') and part.text:
                return part.text
    return ""


def get_api_key(model_name: Optional[str] = None) -> str:
    """
    Return the API key for your chosen LLM provider.
    Uses read_api_key to fetch from the configured API key file.

    Args:
        model_name: Optional model name to determine which API key to use

    Returns:
        The appropriate API key
    """
    # Determine which API key to use based on model name
    if model_name and (
        model_name.startswith("gpt") or
        model_name.startswith("o1") or
        model_name.startswith("o3") or
        model_name.startswith("chatgpt")
    ):
        # OpenAI model
        return read_api_key(key_name="openai_api_key")
    else:
        # Default to Gemini
        return read_api_key()




def call_batch_gen_ai(
    prompts: Iterable[str],
    api_key: str,
    model: Optional[str] = "gemini-2.5-flash",
    poll_seconds: int = 10,
) -> List[str]:
    """
    Inline Gemini Batch: no files, small payloads.
    Compatible with google-genai >= 1.48.0.
    """
    client = genai.Client(api_key=api_key)

    model_name = model or "gemini-2.5-flash"
    if not model_name.startswith("models/"):
        model_name = f"models/{model_name}"

    # Each item is a full GenerateContentRequest
    inline_requests = [
        {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": str(p)}],
                }
            ],
            # Optional per-request config:
            # "config": {"temperature": 0.2, "response_modalities": ["text"]},
        }
        for p in prompts
    ]

    # Create batch with inline requests
    job = client.batches.create(model=model_name, src=inline_requests)
    job_name = job.name
    # print(f"Batch created: {job_name}")

    # Poll
    terminal = {
        "JOB_STATE_SUCCEEDED",
        "JOB_STATE_FAILED",
        "JOB_STATE_CANCELLED",
        "JOB_STATE_EXPIRED",
    }
    while True:
        st = client.batches.get(name=job_name)
        # state may be an enum-like object; normalize to string
        state = getattr(getattr(st, "state", None), "name", None) or str(getattr(st, "state", ""))
        if state in terminal:
            break
        time.sleep(poll_seconds)

    if state != "JOB_STATE_SUCCEEDED":
        detail = getattr(st, "error", None)
        raise RuntimeError(f"Batch failed: {state}. Detail: {detail}")

    # Inline results come back here in-order
    dest = getattr(st, "dest", None) or getattr(st, "response", None)
    inlined = getattr(dest, "inlined_responses", None) or getattr(dest, "inlinedResponses", None)
    if not inlined:
        raise ValueError("No inline responses found; ensure you passed src=[...] (inline list).")

    out: List[str] = []
    for r in inlined:
        # Per-item error?
        if getattr(r, "error", None):
            raise RuntimeError(f"Per-request error: {r.error}")

        resp = getattr(r, "response", None)
        if not resp:
            out.append("")
            continue

        # 1) Try resp.text
        txt = getattr(resp, "text", None)
        if txt:
            out.append(txt)
            continue

        # 2) Try candidates[0].content.parts[].text
        cands = getattr(resp, "candidates", None)
        if cands:
            cand0 = cands[0]
            content = getattr(cand0, "content", None)
            parts = getattr(content, "parts", None) if content else None
            if parts:
                out.append("".join(getattr(p, "text", "") for p in parts if getattr(p, "text", None)))
                continue

        # 3) Try parts directly on resp
        parts = getattr(resp, "parts", None)
        if parts:
            out.append("".join(getattr(p, "text", "") for p in parts if getattr(p, "text", None)))
            continue

        out.append("")

    # Optional: warn if any failed inside the batch
    stats = getattr(st, "batch_stats", None)
    if stats and getattr(stats, "failed_request_count", 0):
        print(f"Warning: {stats.failed_request_count} request(s) failed in the batch.")

    return out