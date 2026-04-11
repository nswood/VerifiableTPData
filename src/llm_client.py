from __future__ import annotations

from typing import Dict, Any, Optional, List, Union

# Import genai module for API calls
try:
    import genai  # type: ignore
except ImportError:
    # Fallback if genai module not available
    genai = None  # type: ignore


def generate_problem(
    model: str,
    api_key: str,
    system_prompt: str,
    topic_entry: Dict[str, Any],
    seed: Optional[int] = None,
    previous_descriptions: Optional[List[str]] = None,
    return_prompt: bool = False,
) -> Union[str, tuple[str, str]]:
    """
    Generate a problem using the LLM API.
    Uses genai.call_gen_ai() to make the actual API call.
    
    Args:
        model: Model name to use
        api_key: API key for the LLM service
        system_prompt: System prompt template
        topic_entry: Dictionary containing topic information
        seed: Optional random seed
        previous_descriptions: Optional list of previous problem descriptions from the same topic
        return_prompt: If True, return both response and full prompt as a tuple
        
    Returns:
        If return_prompt is False, returns just the response string.
        If return_prompt is True, returns a tuple of (response, full_prompt).
    """
    if genai is None:
        raise ImportError("genai module not available. Please ensure genai.py exists.")
    
    # Build the user prompt with topic information
    topic_title = topic_entry.get("topic", "")
    topic_description = topic_entry.get("description", "")
    
    user_prompt = f"Generate a new QFT exercise in this format for the topic:\n\"{topic_title}\""
    if topic_description:
        user_prompt += f"\n\nTopic description: {topic_description}"
    
    # Add previous problem descriptions if provided
    if previous_descriptions:
        user_prompt += "\n\nPreviously generated problems in this topic:"
        for i, desc in enumerate(previous_descriptions, 1):
            user_prompt += f"\n{i}. {desc}"
        user_prompt += "\n\nIMPORTANT: Generate a problem with tasks that are different from those listed above. Some similarity is acceptable, but the problem should not be an exact copy or nearly identical to any of the previous problems."
    
    # Combine system prompt and user prompt
    # For Gemini, we can pass system instructions separately if using the newer API
    # For now, combine them in the prompt
    full_prompt = f"{system_prompt}\n\n{user_prompt}"
    
    # Call the Gemini API
    response = genai.call_gen_ai(full_prompt, api_key, model=model)
    
    if return_prompt:
        return response, full_prompt
    return response


def generate_adapted_problem(
    model: str,
    api_key: str,
    system_prompt: str,
    seed_problem: str,
    seed_solution: str,
    return_prompt: bool = False,
) -> Union[str, tuple[str, str]]:
    """
    Adapt a seed problem-solution pair into the target format via the LLM API.
    The LLM reformats the seed problem according to the system prompt format while
    preserving the original problem content.
    
    Args:
        model: Model name to use
        api_key: API key for the LLM service
        system_prompt: System prompt template (defines the output format)
        seed_problem: Original problem text from seed data
        seed_solution: Original solution text from seed data
        return_prompt: If True, return both response and full prompt as a tuple
        
    Returns:
        If return_prompt is False, returns just the response string.
        If return_prompt is True, returns a tuple of (response, full_prompt).
    """
    if genai is None:
        raise ImportError("genai module not available. Please ensure genai.py exists.")
    
    # Ensure seed_problem and seed_solution are strings
    if not isinstance(seed_problem, str):
        if isinstance(seed_problem, dict):
            seed_problem = str(seed_problem)
        else:
            seed_problem = str(seed_problem) if seed_problem else ""
    
    if not isinstance(seed_solution, str):
        if isinstance(seed_solution, dict):
            seed_solution = str(seed_solution)
        else:
            seed_solution = str(seed_solution) if seed_solution else ""
    
    # Build the user prompt asking LLM to reformat the seed problem
    user_prompt = """Reformat the following problem and solution into the format specified in the system prompt above.

Original Problem:
"""
    user_prompt += seed_problem
    user_prompt += "\n\nOriginal Solution:\n"
    user_prompt += seed_solution
    user_prompt += """

Instructions:
- Follow the format specified in the system prompt exactly
- Preserve the original problem content and solution approach
- Convert the problem into the required format with appropriate sections
- Ensure tasks produce evaluable outputs (numeric or categorical)
- Extract or create tasks that match the original problem's intent
- Slight modifications are acceptable to fit the format, but maintain the core problem and solution

Generate the reformatted problem now:"""

    # Combine system prompt and user prompt
    full_prompt = f"{system_prompt}\n\n{user_prompt}"

    # Call the Gemini API
    response = genai.call_gen_ai(full_prompt, api_key, model=model)

    if return_prompt:
        return response, full_prompt
    return response


# Backwards compatibility alias
generate_semi_synthetic_problem = generate_adapted_problem
