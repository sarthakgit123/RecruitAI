"""
chatbot_service.py

Takes retrieval result from chatbot_retrieval.py and generates natural-language answers using OpenRouter LLM.
Chat history is now persisted in PostgreSQL (chat_messages table) instead of an in-memory list.
"""

import time
from typing import Dict, Any, List, Optional
from openai import OpenAI
from sqlalchemy.orm import Session

from app.core.config import settings
from app.services.chatbot.chatbot_retrieval import retrieve
from app.prompts import load_prompt
from app.database.models import ChatMessage


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


def _get_chat_history(db: Session, session_id: str = "default") -> List[Dict[str, str]]:
    """Read chat history from PostgreSQL chat_messages table."""
    messages = (
        db.query(ChatMessage)
        .filter(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.created_at.asc())
        .all()
    )
    return [{"role": msg.role, "content": msg.content} for msg in messages]


def _save_chat_message(db: Session, role: str, content: str, session_id: str = "default") -> None:
    """Save a single chat message to PostgreSQL."""
    msg = ChatMessage(session_id=session_id, role=role, content=content)
    db.add(msg)
    db.commit()


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
    history_text = _format_history_for_prompt(history or [])

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


def ask(question: str, db: Session, session_id: str = "default", top_k: int = 4) -> str:
    """
    Main chatbot entry point.
    
    Args:
        question: The user's question.
        db: SQLAlchemy database session (injected by FastAPI Depends).
        session_id: Chat session identifier for multi-user support.
        top_k: Number of top candidate chunks to retrieve.
    
    Returns:
        The generated answer string.
    """
    retrieval_result = retrieve(question, top_k=top_k)

    # Load conversation history from PostgreSQL
    history = _get_chat_history(db, session_id=session_id)
    answer = generate_answer(question, retrieval_result, history=history)

    # Save both user question and assistant answer to PostgreSQL
    _save_chat_message(db, role="user", content=question, session_id=session_id)
    _save_chat_message(db, role="assistant", content=answer, session_id=session_id)

    return answer


def reset_history(db: Session, session_id: str = "default") -> None:
    """Delete all chat messages for a session from PostgreSQL."""
    db.query(ChatMessage).filter(ChatMessage.session_id == session_id).delete()
    db.commit()
