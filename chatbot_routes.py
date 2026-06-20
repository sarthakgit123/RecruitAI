"""
chatbot_routes.py

FastAPI routes for the resume chatbot. Kept in a separate file and
included into the main app, rather than crammed into main.py, so the
existing upload/match routes stay untouched and this can be reviewed
independently.

Exposes:
- GET  /chat          -> renders the chat page (chat.html)
- POST /chat-ui       -> form-based endpoint for the chat page itself,
                          returns an HTML fragment (for use with HTMX
                          or a simple form post + page reload)
- POST /chat-api      -> JSON endpoint for AJAX/fetch-based frontends
- POST /chat-reset    -> clears chat history for a fresh session
"""

from fastapi import APIRouter, Request, Form
from fastapi.templating import Jinja2Templates

from services.Chatbot_service import ask, reset_history

router = APIRouter()
templates = Jinja2Templates(directory="templates")


@router.get("/chat")
async def chat_page(request: Request):
    """Renders the initial chat page with an empty conversation."""
    return templates.TemplateResponse(
        request=request,
        name="chat.html",
        context={"title": "Resume Chatbot", "messages": []}
    )


@router.post("/chat-api")
async def chat_api(question: str = Form(...)):
    """
    JSON endpoint - takes a question, returns the answer as JSON.
    Use this if the frontend is doing fetch()/AJAX calls (the
    chat.html provided alongside this uses this endpoint).
    """
    try:
        answer = ask(question)
        return {"status": "success", "answer": answer}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@router.post("/chat-reset")
async def chat_reset():
    """Clears the in-memory chat history, starting a fresh session."""
    reset_history()
    return {"status": "success", "message": "Chat history cleared"}