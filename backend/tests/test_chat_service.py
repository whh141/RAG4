"""Unit tests for ChatService stream completion guarantees."""

from __future__ import annotations

import sys
import types
import unittest
from typing import Any

if "fastapi" not in sys.modules:
    fastapi_stub = types.ModuleType("fastapi")

    class HTTPException(Exception):
        def __init__(self, status_code: int, detail: str) -> None:
            super().__init__(detail)
            self.status_code = status_code
            self.detail = detail

    class _Status:
        HTTP_400_BAD_REQUEST = 400
        HTTP_500_INTERNAL_SERVER_ERROR = 500

    fastapi_stub.HTTPException = HTTPException
    fastapi_stub.status = _Status()
    sys.modules["fastapi"] = fastapi_stub
else:
    from fastapi import HTTPException  # type: ignore[no-redef]

if "app.core.graph" not in sys.modules:
    graph_stub = types.ModuleType("app.core.graph")

    class _PlaceholderGraph:
        async def astream_events(self, _state_input: dict[str, Any], version: str):
            if version != "v2":
                raise ValueError("unexpected stream version")
            if False:
                yield {}

    graph_stub.GRAPH = _PlaceholderGraph()
    sys.modules["app.core.graph"] = graph_stub

if "app.core.state" not in sys.modules:
    state_stub = types.ModuleType("app.core.state")

    class IntentType:
        LOCAL = "LOCAL"

    state_stub.IntentType = IntentType
    sys.modules["app.core.state"] = state_stub

if "app.db.sqlite_mgr" not in sys.modules:
    sqlite_stub = types.ModuleType("app.db.sqlite_mgr")

    class SQLiteManager:  # pragma: no cover - placeholder for import typing
        pass

    sqlite_stub.SQLiteManager = SQLiteManager
    sys.modules["app.db.sqlite_mgr"] = sqlite_stub

from app.services import chat_service as chat_service_module
from app.services.chat_service import ChatService


class FakeSQLiteManager:
    def __init__(self) -> None:
        self.messages: list[tuple[str, str, str]] = []

    def ensure_session(self, session_id: str) -> None:
        return None

    def append_message(self, session_id: str, role: str, content: str) -> None:
        self.messages.append((session_id, role, content))

    def get_messages(self, session_id: str) -> list[Any]:
        class Row:
            def __init__(self, role: str, content: str) -> None:
                self.role = role
                self.content = content

        return [Row(role, content) for sid, role, content in self.messages if sid == session_id]


class FakeGraph:
    def __init__(self, events: list[dict[str, Any]]) -> None:
        self._events = events

    async def astream_events(self, _state_input: dict[str, Any], version: str):
        if version != "v2":
            raise ValueError("unexpected stream version")
        for event in self._events:
            yield event


class ChatServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_emit_final_answer_when_no_stream_tokens(self) -> None:
        sqlite_mgr = FakeSQLiteManager()
        service = ChatService(sqlite_mgr=sqlite_mgr)

        chat_service_module.GRAPH = FakeGraph(
            [
                {
                    "event": "on_chain_end",
                    "metadata": {"langgraph_node": "cot_generate_node"},
                    "data": {"output": {"final_answer": "<think>a</think><answer>b</answer>"}},
                }
            ]
        )

        packets = [packet async for packet in service.stream_chat("s1", "hello")]

        self.assertEqual(packets, [{"type": "token", "content": "<think>a</think><answer>b</answer>"}])
        self.assertEqual(sqlite_mgr.messages[-1], ("s1", "assistant", "<think>a</think><answer>b</answer>"))

    async def test_emit_remaining_suffix_when_partial_tokens_streamed(self) -> None:
        sqlite_mgr = FakeSQLiteManager()
        service = ChatService(sqlite_mgr=sqlite_mgr)

        chat_service_module.GRAPH = FakeGraph(
            [
                {
                    "event": "on_chat_model_stream",
                    "metadata": {"langgraph_node": "cot_generate_node"},
                    "data": {"chunk": type("Chunk", (), {"content": "<think>a</think>"})()},
                },
                {
                    "event": "on_chain_end",
                    "metadata": {"langgraph_node": "cot_generate_node"},
                    "data": {"output": {"final_answer": "<think>a</think><answer>b</answer>"}},
                },
            ]
        )

        packets = [packet async for packet in service.stream_chat("s2", "hello")]

        self.assertEqual(
            packets,
            [
                {"type": "token", "content": "<think>a</think>"},
                {"type": "token", "content": "<answer>b</answer>"},
            ],
        )

    async def test_raise_on_stream_and_final_mismatch(self) -> None:
        sqlite_mgr = FakeSQLiteManager()
        service = ChatService(sqlite_mgr=sqlite_mgr)

        chat_service_module.GRAPH = FakeGraph(
            [
                {
                    "event": "on_chat_model_stream",
                    "metadata": {"langgraph_node": "cot_generate_node"},
                    "data": {"chunk": type("Chunk", (), {"content": "abc"})()},
                },
                {
                    "event": "on_chain_end",
                    "metadata": {"langgraph_node": "cot_generate_node"},
                    "data": {"output": {"final_answer": "xyz"}},
                },
            ]
        )

        with self.assertRaises(HTTPException):
            _ = [packet async for packet in service.stream_chat("s3", "hello")]


if __name__ == "__main__":
    unittest.main()
