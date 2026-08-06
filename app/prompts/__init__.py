"""
prompts package

Centralized LLM prompt templates for RecruitAI.
All prompt strings are stored as .txt files in this directory and loaded at runtime.
"""

import os
from pathlib import Path
from functools import lru_cache

PROMPTS_DIR = Path(__file__).resolve().parent


@lru_cache(maxsize=None)
def load_prompt(name: str) -> str:
    """
    Load a prompt template from a .txt file in the prompts directory.
    
    Args:
        name: The filename (without extension) of the prompt template.
    
    Returns:
        The prompt template string.
    
    Raises:
        FileNotFoundError: If the prompt template file does not exist.
    """
    filepath = PROMPTS_DIR / f"{name}.txt"
    if not filepath.exists():
        raise FileNotFoundError(f"Prompt template not found: {filepath}")
    return filepath.read_text(encoding="utf-8")
