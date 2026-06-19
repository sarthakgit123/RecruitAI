"""
chatbot_service.py

The final piece of the RAG pipeline: takes the retrieval result from
chatbot_retrieval.py (matched candidates + relevant chunks) and asks
Gemini to compose an actual natural-language answer, grounded only in
the retrieved context - not the model's general knowledge.

Also handles simple conversation history, so follow-up questions like
"what about his projects?" can resolve who "his" refers to.

Usage:
    from chatbot_service import ask

    answer = ask("Who knows Python?")
    print(answer)

    # Follow-up in the same session reuses history automatically
    answer2 = ask("Which of them has the most experience?")
"""

import os
import time
import json
from google import genai
from google.genai import errors as genai_errors
from dotenv import load_dotenv

from chatbot_retrieval import retrieve

load_dotenv()

MAX_RETRIES = 3

# Simple in-memory chat history for this process. Each entry is
# {"role": "user"/"assistant", "content": str}. Good enough for a
# single chat session (e.g. one Streamlit/CLI run) - if you need
# multi-user or persistent history later, this is the place to swap
# in a database-backed store instead of a plain list.
_chat_history = []


def _call_gemini_with_retry(prompt):
    """
    Shared retry wrapper for Gemini calls - same transient-503
    handling used in chatbot_retrieval.py, kept here too since this
    is a separate call to the same flaky endpoint.
    """
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    for attempt in range(MAX_RETRIES):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )
            return response.text.strip()
        except genai_errors.ServerError as e:
            if attempt < MAX_RETRIES - 1:
                wait_seconds = 2 ** attempt
                print(f"Gemini busy, retrying in {wait_seconds}s...")
                time.sleep(wait_seconds)
            else:
                print(f"Gemini unavailable after {MAX_RETRIES} attempts: {e}")
                return None

    return None


def _format_chunks_for_prompt(chunks):
    """
    Turns the retrieved chunk list into readable text for the prompt,
    grouped by candidate so the LLM doesn't have to mentally regroup
    scattered chunks itself. Each candidate's chunks appear together
    under their name.
    """
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


def _format_history_for_prompt(history, max_turns=6):
    """
    Includes only the last few turns to keep the prompt small - older
    context rarely matters for a resume-search chat and bloating the
    prompt costs tokens for no benefit.
    """
    if not history:
        return ""

    recent = history[-max_turns:]
    lines = [f"{turn['role']}: {turn['content']}" for turn in recent]
    return "\n".join(lines)


def generate_answer(question, retrieval_result, history=None):
    """
    Builds the final prompt from the question + retrieved context +
    recent history, and asks Gemini to answer.

    Three explicit cases are handled differently in the prompt:
    1. Hard filters were given but matched zero candidates -> tell
       Gemini plainly so it says "no one matches" instead of
       inventing an answer from irrelevant leftover chunks.
    2. Chunks were retrieved -> answer grounded in those chunks only.
    3. (implicit) No hard filters, broad semantic results -> same as
       case 2, just with a wider candidate pool.
    """
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

    prompt = f"""
    You are a helpful assistant answering questions about job candidates
    based ONLY on the resume information provided below. Do not use any
    outside knowledge and do not invent details that aren't present in
    the context.

    Rules:
    - Answer using ONLY the candidate information given below.
    - If the context doesn't fully answer the question, say so plainly
      rather than guessing.
    - When multiple candidates are relevant, mention all of them and
      compare/list them clearly - do not focus on just one candidate
      unless the question is specifically about one person.
    - Refer to candidates by name.
    - Be concise and direct.

    {f"Recent conversation:\n{history_text}\n" if history_text else ""}
    Candidate information retrieved for this question:
    {context_text}

    Question: {question}

    Answer:
    """

    answer = _call_gemini_with_retry(prompt)

    if answer is None:
        return (
            "I couldn't reach the AI service to generate an answer right now "
            "(it's temporarily overloaded). Please try again in a moment."
        )

    return answer


def ask(question, top_k=8, use_history=True):
    """
    Main entry point for the chatbot. Runs retrieval, generates an
    answer, and updates the in-memory chat history.
    """
    retrieval_result = retrieve(question, top_k=top_k)

    history = _chat_history if use_history else []
    answer = generate_answer(question, retrieval_result, history=history)

    if use_history:
        _chat_history.append({"role": "user", "content": question})
        _chat_history.append({"role": "assistant", "content": answer})

    return answer


def reset_history():
    """Clears the in-memory chat history - call this to start a fresh session."""
    global _chat_history
    _chat_history = []


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        q = " ".join(sys.argv[1:])
        print(ask(q))
    else:
        print("Resume chatbot - type 'exit' to quit.\n")
        while True:
            q = input("You: ").strip()
            if q.lower() in ("exit", "quit"):
                break
            if not q:
                continue
            answer = ask(q)
            print(f"\nBot: {answer}\n")