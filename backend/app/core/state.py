"""LangGraph state definitions for the Agentic RAG workflow."""

from __future__ import annotations

from enum import Enum
from typing import TypedDict


class IntentType(str, Enum):
    """Supported intent routing categories."""

    LOCAL = "LOCAL"
    WEB = "WEB"
    COMPLEX = "COMPLEX"


class ChatMessage(TypedDict):
    """Normalized persisted chat message format."""

    role: str
    content: str


class RetrievedDocument(TypedDict, total=False):
    """Unified retrieval item structure from local vector DB or web search."""

    source_type: str
    source: str
    title: str
    content: str
    score: float
    url: str


class AgentState(TypedDict):
    """State container passed between all graph nodes."""

    session_id: str
    chat_history: list[ChatMessage]
    user_query: str
    standalone_query: str
    intent_type: IntentType
    retrieved_docs: list[RetrievedDocument]
    filtered_facts: str
    final_answer: str


__all__ = [
    "IntentType",
    "ChatMessage",
    "RetrievedDocument",
    "AgentState",
]
