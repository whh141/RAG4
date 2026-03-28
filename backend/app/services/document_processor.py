"""Document processing service for KB uploads and dual-write persistence."""

from __future__ import annotations

from pathlib import Path

from fastapi import HTTPException, UploadFile, status
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document as LCDocument

from app.db.chroma_mgr import ChromaManager
from app.db.sqlite_mgr import SQLiteManager


class DocumentProcessor:
    """Coordinates upload parsing, chunking, and dual-write persistence."""

    def __init__(
        self,
        sqlite_mgr: SQLiteManager,
        chroma_mgr: ChromaManager,
        upload_dir: str = "backend/data/uploads",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> None:
        self.sqlite_mgr = sqlite_mgr
        self.chroma_mgr = chroma_mgr
        self.upload_path = Path(upload_dir)
        self.upload_path.mkdir(parents=True, exist_ok=True)
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )

    async def process_upload(self, upload_file: UploadFile) -> dict[str, str | int]:
        """Persist file, chunk content, and execute SQLite+Chroma dual-write."""
        if upload_file.filename is None or not upload_file.filename.strip():
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="filename is required")

        suffix = Path(upload_file.filename).suffix.lower()
        if suffix not in {".pdf", ".txt", ".md", ".markdown"}:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Unsupported file type. Only PDF/TXT/Markdown are allowed.",
            )

        safe_name = Path(upload_file.filename).name
        document = self.sqlite_mgr.create_document(file_name=safe_name, status="processing", chunk_count=0)

        local_name = f"{document.id}_{safe_name}"
        local_file_path = self.upload_path / local_name

        try:
            file_bytes = await upload_file.read()
            local_file_path.write_bytes(file_bytes)

            loader = self._resolve_loader(local_file_path, suffix)
            raw_docs = loader.load()
            if not raw_docs:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail="Uploaded file does not contain readable text.",
                )

            split_docs = self._split_documents(raw_docs, document.id, safe_name)
            chunks = [doc.page_content for doc in split_docs]
            metadatas = [doc.metadata for doc in split_docs]

            self.chroma_mgr.add_chunks(document_id=document.id, chunks=chunks, metadatas=metadatas)
            self.sqlite_mgr.update_document_status(
                document.id,
                status="completed",
                chunk_count=len(split_docs),
            )

            return {
                "file_id": document.id,
                "file_name": safe_name,
                "status": "completed",
                "chunk_count": len(split_docs),
            }
        except Exception:
            self.sqlite_mgr.update_document_status(document.id, status="error", chunk_count=0)
            raise

    def _resolve_loader(self, file_path: Path, suffix: str) -> TextLoader | PyPDFLoader:
        if suffix == ".pdf":
            return PyPDFLoader(str(file_path))
        return TextLoader(str(file_path), encoding="utf-8")

    def _split_documents(
        self,
        raw_docs: list[LCDocument],
        file_id: str,
        file_name: str,
    ) -> list[LCDocument]:
        split_docs = self.splitter.split_documents(raw_docs)
        for idx, doc in enumerate(split_docs):
            merged_metadata = dict(doc.metadata)
            merged_metadata["file_id"] = file_id
            merged_metadata["file_name"] = file_name
            merged_metadata["title"] = file_name
            merged_metadata["chunk_index"] = idx
            doc.metadata = merged_metadata
        return split_docs


def build_document_processor() -> DocumentProcessor:
    """Factory helper for API layer."""
    sqlite_mgr = SQLiteManager()
    chroma_mgr = ChromaManager()
    return DocumentProcessor(sqlite_mgr=sqlite_mgr, chroma_mgr=chroma_mgr)


__all__ = ["DocumentProcessor", "build_document_processor"]
