"""ChromaDB manager for vector document storage and retrieval."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.config import Settings


class ChromaManager:
    """Data access wrapper around ChromaDB persistent collections."""

    def __init__(
        self,
        persist_directory: str = "backend/data/chromadb",
        collection_name: str = "kb_documents",
    ) -> None:
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection_name = collection_name
        self.collection = self._get_or_create_collection(collection_name)

    def _get_or_create_collection(self, collection_name: str) -> Collection:
        return self.client.get_or_create_collection(name=collection_name)

    def add_chunks(
        self,
        *,
        document_id: str,
        chunks: list[str],
        metadatas: list[dict[str, Any]] | None = None,
        ids: list[str] | None = None,
    ) -> list[str]:
        """Insert text chunks into Chroma with linked document metadata."""
        if not document_id.strip():
            raise ValueError("document_id must not be empty")
        if not chunks:
            raise ValueError("chunks must not be empty")

        if metadatas is not None and len(metadatas) != len(chunks):
            raise ValueError("metadatas length must equal chunks length")

        chunk_ids = ids if ids is not None else [str(uuid4()) for _ in chunks]
        if len(chunk_ids) != len(chunks):
            raise ValueError("ids length must equal chunks length")

        normalized_metadata: list[dict[str, Any]] = []
        for idx, _ in enumerate(chunks):
            metadata = dict(metadatas[idx]) if metadatas else {}
            metadata["document_id"] = document_id
            metadata["chunk_index"] = idx
            normalized_metadata.append(metadata)

        self.collection.add(documents=chunks, metadatas=normalized_metadata, ids=chunk_ids)
        return chunk_ids

    def query(
        self,
        *,
        query_text: str,
        n_results: int = 5,
        where: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run semantic search over the collection."""
        if not query_text.strip():
            raise ValueError("query_text must not be empty")
        if n_results <= 0:
            raise ValueError("n_results must be greater than zero")

        return self.collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where=where,
        )

    def get_by_document_id(self, document_id: str) -> dict[str, Any]:
        """Fetch all chunks for a specific source document."""
        if not document_id.strip():
            raise ValueError("document_id must not be empty")
        return self.collection.get(where={"document_id": document_id})

    def list_document_ids(self) -> list[str]:
        """List unique document IDs currently represented in the vector store."""
        result = self.collection.get(include=["metadatas"])
        metadatas = result.get("metadatas") or []
        ids: set[str] = set()
        for item in metadatas:
            if isinstance(item, dict) and "document_id" in item:
                ids.add(str(item["document_id"]))
            if isinstance(item, list):
                for nested in item:
                    if isinstance(nested, dict) and "document_id" in nested:
                        ids.add(str(nested["document_id"]))
        return sorted(ids)

    def delete_chunks_by_document_id(self, document_id: str) -> None:
        """Delete all chunks associated with a given document_id."""
        if not document_id.strip():
            raise ValueError("document_id must not be empty")

        existing = self.get_by_document_id(document_id)
        existing_ids = existing.get("ids") or []
        if len(existing_ids) == 0:
            raise ValueError(f"No vectors found for document_id={document_id}")

        self.collection.delete(where={"document_id": document_id})

    def delete_by_chunk_ids(self, ids: Iterable[str]) -> None:
        """Delete vectors by explicit chunk IDs."""
        ids_list = list(ids)
        if not ids_list:
            raise ValueError("ids must not be empty")
        self.collection.delete(ids=ids_list)

    def reset_collection(self) -> None:
        """Hard reset of the active collection."""
        self.client.delete_collection(self.collection_name)
        self.collection = self._get_or_create_collection(self.collection_name)


__all__ = ["ChromaManager"]
