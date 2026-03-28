"""Knowledge base administration REST endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, UploadFile, status

from ..db.chroma_mgr import ChromaManager
from ..db.sqlite_mgr import SQLiteManager
from ..services.document_processor import build_document_processor

router = APIRouter(prefix="/api/kb", tags=["knowledge-base"])

sqlite_mgr = SQLiteManager()
chroma_mgr = ChromaManager()
document_processor = build_document_processor()


@router.post("/upload", status_code=status.HTTP_201_CREATED)
async def upload_document(file: UploadFile) -> dict[str, str | int]:
    try:
        return await document_processor.process_upload(file)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process upload: {exc}",
        ) from exc


@router.get("/files")
def list_documents(
    limit: int = Query(default=20, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> dict[str, object]:
    try:
        total = sqlite_mgr.count_documents()
        paged_rows = sqlite_mgr.list_documents_paginated(limit=limit, offset=offset)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list files: {exc}",
        ) from exc

    return {
        "items": [
            {
                "file_id": row.id,
                "file_name": row.file_name,
                "status": row.status,
                "chunk_count": row.chunk_count,
                "created_at": row.created_at.isoformat(),
                "updated_at": row.updated_at.isoformat(),
            }
            for row in paged_rows
        ],
        "limit": limit,
        "offset": offset,
        "total": total,
    }


@router.delete("/files/{file_id}", status_code=status.HTTP_200_OK)
def delete_document(file_id: str) -> dict[str, str]:
    try:
        document = sqlite_mgr.get_document(file_id)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    try:
        current_status = document.status
        if current_status == "completed":
            sqlite_mgr.update_document_status(file_id, status="deleting")
            current_status = "deleting"

        _delete_kb_record_and_vectors(file_id=file_id, document_status=current_status)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete file {file_id}: {exc}",
        ) from exc

    return {"file_id": file_id, "status": "deleted"}


def _delete_kb_record_and_vectors(*, file_id: str, document_status: str) -> None:
    existing = chroma_mgr.get_by_document_id(file_id)
    existing_ids = existing.get("ids") or []

    if existing_ids:
        chroma_mgr.delete_chunks_by_document_id(file_id)
    elif document_status not in {"error", "processing", "deleting"}:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Vector data missing for unexpected document status={document_status}, file_id={file_id}.",
        )

    sqlite_mgr.delete_document(file_id)


__all__ = ["router"]
