"""Application configuration helpers."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    openai_api_key: str
    openai_organization: str | None
    openai_project: str | None
    gpt_model: str
    embedding_model: str
    database_url: str
    chunk_size: int = 220
    chunk_overlap: int = 40


def load_settings() -> Settings:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY must be set in the environment.")

    primary_database_url = os.getenv("DATABASE_URL")
    sqlalchemy_url = os.getenv("SQLALCHEMY_DATABASE_URL") or os.getenv("SQLALCHEMY_DATABASE_URI")

    if primary_database_url and sqlalchemy_url and primary_database_url != sqlalchemy_url:
        raise RuntimeError(
            "DATABASE_URL and SQLALCHEMY_DATABASE_URL/SQLALCHEMY_DATABASE_URI differ; "
            "set only one to avoid ambiguity."
        )

    database_url = primary_database_url or sqlalchemy_url
    if not database_url:
        sqlite_path = (Path(__file__).resolve().parent / "policy_vectors.db").as_posix()
        database_url = f"sqlite:///{sqlite_path}"

    if not database_url:
        raise RuntimeError(
            "Set DATABASE_URL (Render/Postgres) or SQLALCHEMY_DATABASE_URL/SQLALCHEMY_DATABASE_URI "
            "to enable persistence."
        )

    return Settings(
        openai_api_key=api_key,
        openai_organization=os.getenv("OPENAI_ORGANIZATION"),
        openai_project=os.getenv("OPENAI_PROJECT"),
        gpt_model=os.getenv("OPENAI_COMPLETION_MODEL", "gpt-5-nano"),
        embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-large"),
        database_url=database_url,
        chunk_size=int(os.getenv("CHUNK_SIZE", "220")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "40")),
    )
