"""Answer generation (Groq) and prompt construction."""

from app.generation.groq_client import (
    GroqGenerator,
    extract_json_object,
    get_generator,
    reset_generator,
)
from app.generation.prompts import (
    INSUFFICIENT_EVIDENCE,
    SYSTEM_PROMPT,
    build_messages,
    neutralise_delimiters,
    render_context_block,
)

__all__ = [
    "INSUFFICIENT_EVIDENCE",
    "SYSTEM_PROMPT",
    "GroqGenerator",
    "build_messages",
    "extract_json_object",
    "get_generator",
    "neutralise_delimiters",
    "render_context_block",
    "reset_generator",
]
