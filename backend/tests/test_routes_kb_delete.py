"""Unit tests for KB delete dual-delete consistency rules."""

from __future__ import annotations

import unittest

from fastapi import HTTPException

from app.api import routes_kb


class Doc:
    def __init__(self, doc_id: str, status: str) -> None:
        self.id = doc_id
        self.status = status


class FakeSQLite:
    def __init__(self, doc: Doc) -> None:
        self.doc = doc
        self.deleted: list[str] = []

    def get_document(self, file_id: str) -> Doc:
        if file_id != self.doc.id:
            raise ValueError("Document not found")
        return self.doc

    def delete_document(self, file_id: str) -> None:
        self.deleted.append(file_id)


class FakeChroma:
    def __init__(self, ids: list[str]) -> None:
        self.ids = ids
        self.deleted: list[str] = []

    def get_by_document_id(self, file_id: str):
        return {"ids": list(self.ids)}

    def delete_chunks_by_document_id(self, file_id: str) -> None:
        self.deleted.append(file_id)


class DeleteRouteTests(unittest.TestCase):
    def test_delete_error_document_without_vectors(self) -> None:
        routes_kb.sqlite_mgr = FakeSQLite(Doc("d1", "error"))
        routes_kb.chroma_mgr = FakeChroma([])

        response = routes_kb.delete_document("d1")

        self.assertEqual(response["status"], "deleted")
        self.assertEqual(routes_kb.sqlite_mgr.deleted, ["d1"])
        self.assertEqual(routes_kb.chroma_mgr.deleted, [])

    def test_delete_completed_document_without_vectors_raises(self) -> None:
        routes_kb.sqlite_mgr = FakeSQLite(Doc("d2", "completed"))
        routes_kb.chroma_mgr = FakeChroma([])

        with self.assertRaises(HTTPException):
            routes_kb.delete_document("d2")

        self.assertEqual(routes_kb.sqlite_mgr.deleted, [])

    def test_delete_document_with_vectors_deletes_both(self) -> None:
        routes_kb.sqlite_mgr = FakeSQLite(Doc("d3", "completed"))
        routes_kb.chroma_mgr = FakeChroma(["c1", "c2"])

        response = routes_kb.delete_document("d3")

        self.assertEqual(response["status"], "deleted")
        self.assertEqual(routes_kb.chroma_mgr.deleted, ["d3"])
        self.assertEqual(routes_kb.sqlite_mgr.deleted, ["d3"])


if __name__ == "__main__":
    unittest.main()
