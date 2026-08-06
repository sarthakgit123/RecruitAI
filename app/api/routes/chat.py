"""
chat.py

FastAPI routes for Candidate Chatbot UI and API.
"""

from fastapi import APIRouter, Request, Form
from fastapi.templating import Jinja2Templates

from app.core.config import settings
from app.services.chatbot.chatbot_service import ask, reset_history
from app.schemas.chat import ChatResponse, ChatResetResponse

router = APIRouter()
templates = Jinja2Templates(directory=str(settings.TEMPLATES_DIR))


@router.get("/chat")
async def chat_page(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="chat.html",
        context={"title": "Resume Chatbot", "messages": []}
    )


@router.post("/chat-api", response_model=ChatResponse)
async def chat_api(question: str = Form(...)):
    try:
        answer = ask(question)
        return {"status": "success", "answer": answer}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@router.post("/chat-reset", response_model=ChatResetResponse)
async def chat_reset():
    reset_history()
    return {"status": "success", "message": "Chat history cleared"}
