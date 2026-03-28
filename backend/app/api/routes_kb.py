"""Knowledge base administration REST endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, UploadFile, status

from app.db.chroma_mgr import ChromaManager
from app.db.sqlite_mgr import SQLiteManager
from app.services.document_processor import build_document_processor

router = APIRouter(prefix="/api/kb", tags=["knowledge-base"])

sqlite_mgr = SQLiteManager()
chroma_mgr = ChromaManager()
document_processor = build_document_processor()


@router.post("/upload", status_code=status.HTTP_201_CREATED)
async def upload_document(file: UploadFile) -> dict[str, str | int]:
    """Upload a knowledge file, process it, and perform dual-write persistence."""
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
    """List uploaded documents for admin panel table rendering."""
    try:
        rows = sqlite_mgr.list_documents()
        paged_rows = rows[offset : offset + limit]
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
        "total": len(rows),
    }


@router.delete("/files/{file_id}", status_code=status.HTTP_200_OK)
def delete_document(file_id: str) -> dict[str, str]:
    """Delete KB record and its vectors using strict dual-delete semantics."""
    try:
        document = sqlite_mgr.get_document(file_id)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    try:
        existing = chroma_mgr.get_by_document_id(file_id)
        existing_ids = existing.get("ids") or []

        if existing_ids:
            chroma_mgr.delete_chunks_by_document_id(file_id)
        elif document.status == "completed":
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Vector data missing for completed document {file_id}.",
            )

        sqlite_mgr.delete_document(file_id)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete file {file_id}: {exc}",
        ) from exc

    return {"file_id": file_id, "status": "deleted"}


__all__ = ["router"]
