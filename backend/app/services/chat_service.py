"""Chat service layer managing session lifecycle and LangGraph streaming."""

from __future__ import annotations

from typing import Any, AsyncIterator

from fastapi import HTTPException, status

from app.core.graph import GRAPH
from app.core.state import IntentType
from app.db.sqlite_mgr import SQLiteManager


class ChatService:
    """Service for validating chat input, running graph, and persisting messages."""

    NODE_STATUS_MESSAGES: dict[str, str] = {
        "query_rewrite_node": "Rewriting query with context...",
        "intent_router_node": "Classifying query intent...",
        "local_retrieve_node": "Retrieving from local knowledge base...",
        "web_search_node": "Searching the web for evidence...",
        "complex_reason_node": "Decomposing complex question and retrieving hybrid evidence...",
        "rerank_filter_node": "Filtering and compressing key facts...",
        "cot_generate_node": "Generating final response with reasoning...",
    }

    def __init__(self, sqlite_mgr: SQLiteManager) -> None:
        self.sqlite_mgr = sqlite_mgr

    async def stream_chat(self, session_id: str, user_query: str) -> AsyncIterator[dict[str, str]]:
        """Run graph execution and yield strict status/token packets only."""
        query = user_query.strip()
        if not query:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="query must not be empty")

        self.sqlite_mgr.ensure_session(session_id=session_id)
        self.sqlite_mgr.append_message(session_id=session_id, role="user", content=query)

        history_rows = self.sqlite_mgr.get_messages(session_id=session_id)
        chat_history = [{"role": row.role, "content": row.content} for row in history_rows[:-1]]

        state_input = {
            "session_id": session_id,
            "chat_history": chat_history,
            "user_query": query,
            "standalone_query": "",
            "intent_type": IntentType.LOCAL,
            "retrieved_docs": [],
            "filtered_facts": "",
            "final_answer": "",
        }

        final_answer = ""
        emitted_answer_text = ""

        async for event in GRAPH.astream_events(state_input, version="v2"):
            event_name = str(event.get("event", ""))
            metadata = event.get("metadata") or {}
            data = event.get("data") or {}
            node_name = metadata.get("langgraph_node") or event.get("name")

            if event_name == "on_chain_start" and node_name in self.NODE_STATUS_MESSAGES:
                yield {
                    "type": "status",
                    "node": str(node_name),
                    "message": self.NODE_STATUS_MESSAGES[str(node_name)],
                }

            if event_name == "on_chat_model_stream" and metadata.get("langgraph_node") == "cot_generate_node":
                text = self._extract_chunk_text(data.get("chunk"))
                if text:
                    emitted_answer_text += text
                    yield {"type": "token", "content": text}

            if event_name == "on_chain_end" and node_name == "cot_generate_node":
                extracted = self._extract_final_answer_from_output(data.get("output"))
                if extracted:
                    final_answer = extracted

        if not final_answer:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Graph execution ended without final_answer.",
            )

        if not emitted_answer_text:
            yield {"type": "token", "content": final_answer}
        elif final_answer != emitted_answer_text:
            if not final_answer.startswith(emitted_answer_text):
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Streamed token content does not match final_answer.",
                )
            remaining = final_answer[len(emitted_answer_text) :]
            if remaining:
                yield {"type": "token", "content": remaining}

        self.sqlite_mgr.append_message(session_id=session_id, role="assistant", content=final_answer)

    def _extract_chunk_text(self, chunk_payload: Any) -> str:
        if chunk_payload is None:
            return ""

        content = getattr(chunk_payload, "content", None)
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            pieces: list[str] = []
            for item in content:
                if isinstance(item, str):
                    pieces.append(item)
                elif isinstance(item, dict) and isinstance(item.get("text"), str):
                    pieces.append(item["text"])
            return "".join(pieces)

        if isinstance(chunk_payload, str):
            return chunk_payload
        return ""

    def _extract_final_answer_from_output(self, output: Any) -> str:
        if isinstance(output, dict):
            if isinstance(output.get("final_answer"), str):
                return output["final_answer"].strip()
            for value in output.values():
                extracted = self._extract_final_answer_from_output(value)
                if extracted:
                    return extracted
        return ""


__all__ = ["ChatService"]
