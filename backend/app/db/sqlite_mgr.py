"""SQLite manager for metadata and multi-turn chat persistence."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator, Literal
from uuid import uuid4

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, create_engine, desc, func
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship, sessionmaker


class Base(DeclarativeBase):
    """Base declarative model for SQLAlchemy tables."""


class Document(Base):
    """Stores knowledge-base document metadata."""

    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    file_name: Mapped[str] = mapped_column(String(512), nullable=False, unique=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    chunk_count: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), nullable=False, default=datetime.utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow
    )


class SessionRecord(Base):
    """Conversation session table."""

    __tablename__ = "sessions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    title: Mapped[str] = mapped_column(String(255), nullable=False, default="New Chat")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), nullable=False, default=datetime.utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    messages: Mapped[list["Message"]] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="Message.created_at",
    )


class Message(Base):
    """Multi-turn chat messages linked to a session."""

    __tablename__ = "messages"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    session_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    role: Mapped[Literal["user", "assistant"]] = mapped_column(String(16), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), nullable=False, default=datetime.utcnow, index=True
    )

    session: Mapped[SessionRecord] = relationship(back_populates="messages")


class SQLiteManager:
    """Database access layer for documents, sessions, and messages."""

    def __init__(self, db_path: str = "backend/data/sqlite.db") -> None:
        db_file = Path(db_path)
        db_file.parent.mkdir(parents=True, exist_ok=True)

        self.engine = create_engine(
            f"sqlite:///{db_file}",
            connect_args={"check_same_thread": False},
            future=True,
        )
        self.session_factory = sessionmaker(
            bind=self.engine,
            class_=Session,
            autoflush=False,
            autocommit=False,
            expire_on_commit=False,
        )

    def init_db(self) -> None:
        """Create all required tables if they do not exist."""
        Base.metadata.create_all(self.engine)

    @contextmanager
    def session_scope(self) -> Iterator[Session]:
        """Transactional SQLAlchemy session context manager."""
        session = self.session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    # ---------- Document metadata CRUD ----------
    def create_document(self, file_name: str, status: str, chunk_count: int) -> Document:
        with self.session_scope() as session:
            document = Document(file_name=file_name, status=status, chunk_count=chunk_count)
            session.add(document)
            session.flush()
            session.refresh(document)
            return document

    def update_document_status(
        self,
        document_id: str,
        *,
        status: str,
        chunk_count: int | None = None,
    ) -> Document:
        with self.session_scope() as session:
            document = session.get(Document, document_id)
            if document is None:
                raise ValueError(f"Document not found for id={document_id}")
            document.status = status
            if chunk_count is not None:
                document.chunk_count = chunk_count
            session.flush()
            session.refresh(document)
            return document

    def get_document(self, document_id: str) -> Document:
        with self.session_scope() as session:
            document = session.get(Document, document_id)
            if document is None:
                raise ValueError(f"Document not found for id={document_id}")
            return document

    def get_document_by_filename(self, file_name: str) -> Document:
        with self.session_scope() as session:
            document = session.query(Document).filter(Document.file_name == file_name).one_or_none()
            if document is None:
                raise ValueError(f"Document not found for file_name={file_name}")
            return document

    def list_documents(self) -> list[Document]:
        with self.session_scope() as session:
            return session.query(Document).order_by(desc(Document.created_at)).all()

    def delete_document(self, document_id: str) -> None:
        with self.session_scope() as session:
            document = session.get(Document, document_id)
            if document is None:
                raise ValueError(f"Document not found for id={document_id}")
            session.delete(document)

    # ---------- Session and message CRUD ----------
    def create_session(self, title: str = "New Chat") -> SessionRecord:
        with self.session_scope() as session:
            record = SessionRecord(title=title)
            session.add(record)
            session.flush()
            session.refresh(record)
            return record

    def ensure_session(self, session_id: str, title: str = "New Chat") -> SessionRecord:
        with self.session_scope() as session:
            record = session.get(SessionRecord, session_id)
            if record is None:
                record = SessionRecord(id=session_id, title=title)
                session.add(record)
                session.flush()
                session.refresh(record)
            return record

    def get_session(self, session_id: str) -> SessionRecord:
        with self.session_scope() as session:
            record = session.get(SessionRecord, session_id)
            if record is None:
                raise ValueError(f"Session not found for id={session_id}")
            return record

    def list_sessions(self) -> list[SessionRecord]:
        with self.session_scope() as session:
            return session.query(SessionRecord).order_by(desc(SessionRecord.updated_at)).all()

    def append_message(self, session_id: str, role: Literal["user", "assistant"], content: str) -> Message:
        with self.session_scope() as session:
            if session.get(SessionRecord, session_id) is None:
                raise ValueError(f"Session not found for id={session_id}")

            message = Message(session_id=session_id, role=role, content=content)
            session.add(message)

            # Force session updated_at update on message append.
            session.query(SessionRecord).filter(SessionRecord.id == session_id).update(
                {SessionRecord.updated_at: func.now()}, synchronize_session=False
            )

            session.flush()
            session.refresh(message)
            return message

    def get_messages(self, session_id: str, limit: int | None = None) -> list[Message]:
        with self.session_scope() as session:
            query = (
                session.query(Message)
                .filter(Message.session_id == session_id)
                .order_by(Message.created_at.asc())
            )
            if limit is not None:
                if limit <= 0:
                    raise ValueError("limit must be greater than zero")
                query = query.limit(limit)
            return query.all()

    def delete_session(self, session_id: str) -> None:
        with self.session_scope() as session:
            record = session.get(SessionRecord, session_id)
            if record is None:
                raise ValueError(f"Session not found for id={session_id}")
            session.delete(record)


__all__ = [
    "Base",
    "Document",
    "SessionRecord",
    "Message",
    "SQLiteManager",
]
