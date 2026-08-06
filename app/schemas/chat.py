"""
chat.py

Pydantic schemas for Chatbot endpoints.
"""

from typing import Optional
from pydantic import BaseModel


class ChatResponse(BaseModel):
    status: str
    answer: Optional[str] = None
    message: Optional[str] = None


class ChatResetResponse(BaseModel):
    status: str
    message: str
