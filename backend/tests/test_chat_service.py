"""Unit tests for ChatService stream completion guarantees."""

from __future__ import annotations

import unittest
from typing import Any

from fastapi import HTTPException

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
