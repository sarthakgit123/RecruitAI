"""
chatbot_service.py

Takes retrieval result from chatbot_retrieval.py and generates natural-language answers using OpenRouter LLM.
"""

import os
import time
from typing import Dict, Any, List, Optional
from openai import OpenAI
from app.core.config import settings
from app.services.chatbot.chatbot_retrieval import retrieve
from app.prompts import load_prompt

_chat_history: List[Dict[str, str]] = []


def _call_llm_with_retry(prompt: str) -> Optional[str]:
    client = OpenAI(
        base_url=settings.OPENROUTER_BASE_URL,
        api_key=settings.OPENROUTER_API_KEY
    )

    last_error = None
    for model_name in settings.OPENROUTER_MODELS:
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1500,
                temperature=0.3,
                extra_body={"models": settings.OPENROUTER_MODELS[:3]}
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            last_error = e
            print(f"Chatbot model {model_name} failed: {e}. Trying fallback...")
            time.sleep(1)

    print(f"All OpenRouter models failed: {last_error}")
    return None


def _format_chunks_for_prompt(chunks: List[Dict[str, Any]]) -> str:
    if not chunks:
        return "No relevant information was found for this question."

    by_candidate = {}
    for chunk in chunks:
        name = chunk.get("meta", {}).get("name") or chunk["candidate_id"]
        by_candidate.setdefault(name, []).append(chunk["text"])

    sections = []
    for name, texts in by_candidate.items():
        joined = "\n".join(f"- {t}" for t in texts)
        sections.append(f"Candidate: {name}\n{joined}")

    return "\n\n".join(sections)


def _format_history_for_prompt(history: List[Dict[str, str]], max_turns: int = 3) -> str:
    if not history:
        return ""

    recent = history[-max_turns:]
    lines = [f"{turn['role']}: {turn['content']}" for turn in recent]
    return "\n".join(lines)


def generate_answer(question: str, retrieval_result: Dict[str, Any], history: Optional[List[Dict[str, str]]] = None) -> str:
    if retrieval_result.get("no_hard_filter_matches"):
        intent = retrieval_result["intent"]
        filters_desc = []
        if intent.get("skill"):
            filters_desc.append(f"skill = {intent['skill']}")
        if intent.get("min_years") is not None:
            filters_desc.append(f"minimum years experience = {intent['min_years']}")
        if intent.get("role"):
            filters_desc.append(f"role = {intent['role']}")

        return (
            f"No candidates matched the criteria you asked about ({', '.join(filters_desc)}). "
            "Try loosening the requirement or asking about a different skill/role."
        )

    context_text = _format_chunks_for_prompt(retrieval_result.get("chunks", []))
    history_text = _format_history_for_prompt(history or _chat_history)

    history_block = f"History:\n{history_text}\n" if history_text else ""
    prompt_template = load_prompt("chatbot_answer")
    prompt = prompt_template.format(
        history_block=history_block,
        context_text=context_text,
        question=question
    )

    answer = _call_llm_with_retry(prompt)

    if answer is None:
        return (
            "I couldn't reach the AI service to generate an answer right now "
            "(it's temporarily overloaded). Please try again in a moment."
        )

    return answer


def ask(question: str, top_k: int = 4, use_history: bool = True) -> str:
    retrieval_result = retrieve(question, top_k=top_k)

    history = _chat_history if use_history else []
    answer = generate_answer(question, retrieval_result, history=history)

    if use_history:
        _chat_history.append({"role": "user", "content": question})
        _chat_history.append({"role": "assistant", "content": answer})

    return answer


def reset_history() -> None:
    global _chat_history
    _chat_history = []
