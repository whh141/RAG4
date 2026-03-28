"""Unit tests for KB delete dual-delete consistency rules."""

from __future__ import annotations

import sys
import types
import unittest

fastapi_stub = sys.modules.get("fastapi")
if fastapi_stub is None:
    fastapi_stub = types.ModuleType("fastapi")
    sys.modules["fastapi"] = fastapi_stub


class HTTPException(Exception):
    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


class _Status:
    HTTP_200_OK = 200
    HTTP_201_CREATED = 201
    HTTP_404_NOT_FOUND = 404
    HTTP_500_INTERNAL_SERVER_ERROR = 500


class APIRouter:
    def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
        pass

    def post(self, *args, **kwargs):  # noqa: ANN002, ANN003
        return lambda fn: fn

    def get(self, *args, **kwargs):  # noqa: ANN002, ANN003
        return lambda fn: fn

    def delete(self, *args, **kwargs):  # noqa: ANN002, ANN003
        return lambda fn: fn


def Query(*args, **kwargs):  # noqa: N802, ANN002, ANN003
    return kwargs.get("default")


class UploadFile:  # pragma: no cover - placeholder
    pass


fastapi_stub.HTTPException = HTTPException
fastapi_stub.status = _Status()
fastapi_stub.APIRouter = APIRouter
fastapi_stub.Query = Query
fastapi_stub.UploadFile = UploadFile

if "app.db.chroma_mgr" not in sys.modules:
    chroma_stub = types.ModuleType("app.db.chroma_mgr")

    class ChromaManager:  # pragma: no cover
        pass

    chroma_stub.ChromaManager = ChromaManager
    sys.modules["app.db.chroma_mgr"] = chroma_stub

if "app.db.sqlite_mgr" not in sys.modules:
    sqlite_stub = types.ModuleType("app.db.sqlite_mgr")

    class SQLiteManager:  # pragma: no cover
        pass

    sqlite_stub.SQLiteManager = SQLiteManager
    sys.modules["app.db.sqlite_mgr"] = sqlite_stub

if "app.services.document_processor" not in sys.modules:
    doc_stub = types.ModuleType("app.services.document_processor")

    def build_document_processor():  # pragma: no cover
        class _Processor:
            async def process_upload(self, _file):
                return {}

        return _Processor()

    doc_stub.build_document_processor = build_document_processor
    sys.modules["app.services.document_processor"] = doc_stub

from app.api import routes_kb


class Doc:
    def __init__(self, doc_id: str, status: str) -> None:
        self.id = doc_id
        self.status = status


class FakeSQLite:
    def __init__(self, doc: Doc) -> None:
        self.doc = doc
        self.deleted: list[str] = []
        self.status_updates: list[tuple[str, str]] = []

    def get_document(self, file_id: str) -> Doc:
        if file_id != self.doc.id:
            raise ValueError("Document not found")
        return self.doc

    def delete_document(self, file_id: str) -> None:
        self.deleted.append(file_id)

    def update_document_status(self, file_id: str, *, status: str, chunk_count: int | None = None) -> Doc:
        if file_id != self.doc.id:
            raise ValueError("Document not found")
        self.doc.status = status
        self.status_updates.append((file_id, status))
        return self.doc


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

    def test_delete_completed_document_without_vectors_succeeds(self) -> None:
        routes_kb.sqlite_mgr = FakeSQLite(Doc("d2", "completed"))
        routes_kb.chroma_mgr = FakeChroma([])

        response = routes_kb.delete_document("d2")

        self.assertEqual(response["status"], "deleted")
        self.assertEqual(routes_kb.sqlite_mgr.status_updates, [("d2", "deleting")])
        self.assertEqual(routes_kb.sqlite_mgr.deleted, ["d2"])

    def test_delete_document_with_vectors_deletes_both(self) -> None:
        routes_kb.sqlite_mgr = FakeSQLite(Doc("d3", "completed"))
        routes_kb.chroma_mgr = FakeChroma(["c1", "c2"])

        response = routes_kb.delete_document("d3")

        self.assertEqual(response["status"], "deleted")
        self.assertEqual(routes_kb.chroma_mgr.deleted, ["d3"])
        self.assertEqual(routes_kb.sqlite_mgr.deleted, ["d3"])


if __name__ == "__main__":
    unittest.main()
