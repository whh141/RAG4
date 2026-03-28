"""State definitions for the Agentic RAG workflow."""

from typing import List, Literal, TypedDict

from langchain_core.documents import Document

IntentType = Literal["LOCAL_FACT", "WEB_SEARCH", "COMPLEX_REASONING"]


class GraphState(TypedDict):
    """State dictionary shared by all LangGraph nodes."""

    user_query: str
    intent_type: IntentType
    search_queries: List[str]
    retrieved_docs: List[Document]
    filtered_facts: str
    final_answer: str
