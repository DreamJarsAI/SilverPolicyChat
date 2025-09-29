"""CLI utility to ingest PDF policies, embed them with OpenAI, and persist in Postgres."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable, List

from openai import OpenAI

from config import Settings, load_settings
from policy_processing import PolicyChunk, discover_policy_documents, load_policy_chunks
from policy_store import PolicyStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def batched(iterable: Iterable[str], size: int) -> Iterable[List[str]]:
    batch: List[str] = []
    for item in iterable:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the school policy embedding index.")
    parser.add_argument(
        "--policies-dir",
        type=Path,
        default=Path(__file__).parent / "policies",
        help="Folder containing the source policy PDFs.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Maximum number of words per chunk before overlap is applied (overrides .env).",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=None,
        help="Approximate number of words that overlap between consecutive chunks (overrides .env).",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Drop existing embeddings before inserting new ones.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Number of chunks to embed per OpenAI request.",
    )
    return parser.parse_args()


def embed_chunks(
    client: OpenAI,
    settings: Settings,
    chunks: List[PolicyChunk],
    *,
    batch_size: int,
) -> List[List[float]]:
    texts = [chunk.text for chunk in chunks]
    embeddings: List[List[float]] = []
    for batch in batched(texts, batch_size):
        response = client.embeddings.create(model=settings.embedding_model, input=batch)
        embeddings.extend([record.embedding for record in response.data])
    return embeddings


def main() -> None:
    args = parse_args()
    settings = load_settings()

    chunk_size = args.chunk_size or settings.chunk_size
    overlap = args.overlap or settings.chunk_overlap

    client = OpenAI(
        api_key=settings.openai_api_key,
        organization=settings.openai_organization,
        project=settings.openai_project,
    )

    store = PolicyStore(settings.database_url)

    documents = discover_policy_documents(args.policies_dir)
    if not documents:
        logger.warning("No policy PDFs found in %s", args.policies_dir)
        return

    chunks = load_policy_chunks(documents, chunk_size=chunk_size, overlap=overlap)
    if not chunks:
        logger.warning("No chunks were produced from the input PDFs.")
        return

    logger.info("Embedding %d chunks across %d documents", len(chunks), len(documents))
    embeddings = embed_chunks(client, settings, chunks, batch_size=args.batch_size)
    dimension = len(embeddings[0])

    store.ensure_schema(dimension)
    if args.rebuild:
        logger.info("Clearing existing embeddings before rebuild.")
        store.delete_all()

    store.store_chunks(documents, chunks, embeddings)
    logger.info("Stored %d chunks across %d documents in Postgres.", len(chunks), len(documents))


if __name__ == "__main__":
    main()
