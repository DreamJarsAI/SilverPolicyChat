"""SQLAlchemy-backed storage for policy document chunks and embeddings."""
from __future__ import annotations

import contextlib
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Iterator, List, Sequence

import numpy as np
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
    delete,
    func,
    select,
)
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Mapped, Session, declarative_base, mapped_column, relationship, sessionmaker
from sqlalchemy.types import TypeDecorator

from policy_processing import PolicyChunk, PolicyDocument


Base = declarative_base()


class FlexibleVector(TypeDecorator[List[float]]):
    """Store embeddings compatibly across SQLite (JSON) and Postgres (pgvector)."""

    impl = Text
    cache_ok = True

    def load_dialect_impl(self, dialect):  # type: ignore[override]
        if dialect.name == "postgresql":
            try:
                from pgvector.sqlalchemy import Vector  # type: ignore[import-not-found]
            except ImportError as exc:  # pragma: no cover - runtime guard
                raise RuntimeError("pgvector is required for Postgres persistence") from exc
            return Vector()
        return dialect.type_descriptor(Text())

    def process_bind_param(self, value: Sequence[float] | None, dialect):
        if value is None:
            return None
        floats = [float(x) for x in value]
        if dialect.name == "postgresql":
            return floats
        return json.dumps(floats)

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        if dialect.name == "postgresql":
            return [float(x) for x in value]
        if isinstance(value, (bytes, bytearray)):
            value = value.decode("utf-8")
        return [float(x) for x in json.loads(value)]


class EmbeddingMetadata(Base):
    __tablename__ = "embedding_metadata"

    key: Mapped[str] = mapped_column(String, primary_key=True)
    value: Mapped[int] = mapped_column(Integer, nullable=False)


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    document_identifier: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    title: Mapped[str] = mapped_column(String, nullable=False)

    chunks: Mapped[List[Chunk]] = relationship("Chunk", back_populates="document", cascade="all, delete-orphan")


class Chunk(Base):
    __tablename__ = "chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    document_id: Mapped[int] = mapped_column(Integer, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    chunk_id: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    page_number: Mapped[int | None] = mapped_column(Integer, nullable=True)
    chunk_index: Mapped[int | None] = mapped_column(Integer, nullable=True)

    document: Mapped[Document] = relationship("Document", back_populates="chunks")
    embedding: Mapped[Embedding | None] = relationship(
        "Embedding",
        back_populates="chunk",
        cascade="all, delete-orphan",
        uselist=False,
    )


class Embedding(Base):
    __tablename__ = "embeddings"

    chunk_id: Mapped[str] = mapped_column(
        String,
        ForeignKey("chunks.chunk_id", ondelete="CASCADE"),
        primary_key=True,
    )
    embedding: Mapped[List[float]] = mapped_column("embedding", FlexibleVector, nullable=False)

    chunk: Mapped[Chunk] = relationship("Chunk", back_populates="embedding")


class UserAccount(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    email: Mapped[str] = mapped_column(String(320), unique=True, nullable=False, index=True)
    password_hash: Mapped[str | None] = mapped_column(String(512), nullable=True)
    is_verified: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    verification_hash: Mapped[str | None] = mapped_column(String(512), nullable=True)
    verification_sent_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    reset_code_hash: Mapped[str | None] = mapped_column(String(512), nullable=True)
    reset_requested_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

@dataclass
class RetrievedChunk:
    title: str
    page_number: int | None
    chunk_index: int | None
    text: str


class PolicyStore:
    """Persist chunks and embeddings using SQLAlchemy."""

    def __init__(self, dsn: str, *, echo: bool = False) -> None:
        connect_args = {"check_same_thread": False} if dsn.startswith("sqlite") else {}
        self.engine: Engine = create_engine(dsn, future=True, echo=echo, connect_args=connect_args)
        self._Session = sessionmaker(bind=self.engine, expire_on_commit=False, autoflush=False, future=True)

    @contextlib.contextmanager
    def session(self) -> Iterator[Session]:
        session = self._Session()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def close(self) -> None:
        self.engine.dispose()

    def ensure_schema(self, embedding_dimension: int | None = None) -> None:
        Base.metadata.create_all(self.engine)
        with self.session() as session:
            record = session.get(EmbeddingMetadata, "embedding_dimension")
            if record is None:
                if embedding_dimension is None:
                    raise RuntimeError(
                        "Embedding dimension is unknown. Build the index first or provide a value."
                    )
                session.add(EmbeddingMetadata(key="embedding_dimension", value=embedding_dimension))
            elif embedding_dimension is not None and record.value != embedding_dimension:
                raise RuntimeError(
                    f"Existing embedding dimension {record.value} does not match {embedding_dimension}."
                )

    def _upsert_documents(self, session: Session, documents: Sequence[PolicyDocument]) -> dict[str, int]:
        mapping: dict[str, int] = {}
        for doc in documents:
            existing = session.execute(
                select(Document).where(Document.document_identifier == doc.document_id)
            ).scalar_one_or_none()
            if existing:
                existing.title = doc.title
                session.flush()
                mapping[doc.document_id] = existing.id
                continue

            new_doc = Document(document_identifier=doc.document_id, title=doc.title)
            session.add(new_doc)
            session.flush()
            mapping[doc.document_id] = new_doc.id
        return mapping

    def store_chunks(
        self,
        documents: Sequence[PolicyDocument],
        chunks: Sequence[PolicyChunk],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("Embeddings and chunk counts do not match.")

        with self.session() as session:
            doc_ids = self._upsert_documents(session, documents)

            for chunk, embedding in zip(chunks, embeddings):
                document_db_id = doc_ids[chunk.document_id]
                existing_chunk = session.execute(
                    select(Chunk).where(Chunk.chunk_id == chunk.chunk_id)
                ).scalar_one_or_none()

                if existing_chunk is None:
                    existing_chunk = Chunk(
                        document_id=document_db_id,
                        chunk_id=chunk.chunk_id,
                        content=chunk.text,
                        page_number=chunk.page_number,
                        chunk_index=chunk.chunk_index,
                    )
                    session.add(existing_chunk)
                else:
                    existing_chunk.document_id = document_db_id
                    existing_chunk.content = chunk.text
                    existing_chunk.page_number = chunk.page_number
                    existing_chunk.chunk_index = chunk.chunk_index

                vector = [float(x) for x in embedding]
                existing_embedding = session.get(Embedding, chunk.chunk_id)
                if existing_embedding is None:
                    session.add(Embedding(chunk_id=chunk.chunk_id, embedding=vector))
                else:
                    existing_embedding.embedding = vector

    def list_documents(self) -> List[str]:
        with self._Session() as session:
            stmt = select(Document.title).order_by(Document.title)
            return [row[0] for row in session.execute(stmt)]

    def similar_chunks(self, embedding: Sequence[float], top_k: int = 4) -> List[RetrievedChunk]:
        query_embedding = np.array(embedding, dtype=float)
        query_norm = float(np.linalg.norm(query_embedding)) or 1.0

        with self._Session() as session:
            stmt = (
                select(
                    Document.title,
                    Chunk.page_number,
                    Chunk.chunk_index,
                    Chunk.content,
                    Embedding.embedding,
                )
                .join(Chunk, Chunk.document_id == Document.id)
                .join(Embedding, Embedding.chunk_id == Chunk.chunk_id)
            )
            rows = session.execute(stmt).all()

        scored: list[tuple[float, RetrievedChunk]] = []
        for title, page_number, chunk_index, content, stored_vector in rows:
            stored_array = np.array(stored_vector, dtype=float)
            denom = float(np.linalg.norm(stored_array)) * query_norm
            if denom == 0.0:
                similarity = 0.0
            else:
                similarity = float(np.dot(query_embedding, stored_array) / denom)
            scored.append(
                (
                    similarity,
                    RetrievedChunk(
                        title=title,
                        page_number=page_number,
                        chunk_index=chunk_index,
                        text=content,
                    ),
                )
            )

        scored.sort(key=lambda item: item[0], reverse=True)
        return [item[1] for item in scored[:top_k]]

    def delete_all(self) -> None:
        with self.session() as session:
            session.execute(delete(Embedding))
            session.execute(delete(Chunk))
            session.execute(delete(Document))
