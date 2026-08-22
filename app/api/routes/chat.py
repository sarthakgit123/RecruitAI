"""
chat.py

FastAPI routes for Candidate Chatbot UI and API.
Chat history is persisted in PostgreSQL via the db session dependency.
"""

from fastapi import APIRouter, Request, Form, Depends
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from app.core.config import settings
from app.services.chatbot.chatbot_service import ask, reset_history
from app.schemas.chat import ChatResponse, ChatResetResponse
from app.database.database import get_db

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
async def chat_api(question: str = Form(...), db: Session = Depends(get_db)):
    try:
        answer = ask(question, db=db)
        return {"status": "success", "answer": answer}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@router.post("/chat-reset", response_model=ChatResetResponse)
async def chat_reset(db: Session = Depends(get_db)):
    reset_history(db=db)
    return {"status": "success", "message": "Chat history cleared"}
