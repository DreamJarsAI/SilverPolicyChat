"""PDF ingestion helpers: discover documents and chunk them for embedding."""
from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Sequence

import pdfplumber

logger = logging.getLogger(__name__)


@dataclass
class PolicyDocument:
    """Metadata about a PDF policy document."""

    document_id: str
    title: str
    path: Path


@dataclass
class PolicyChunk:
    """A single cleaned and sentence-aware chunk extracted from a policy PDF."""

    chunk_id: str
    document_id: str
    title: str
    page_number: int
    chunk_index: int
    text: str


def discover_policy_documents(policies_dir: Path) -> List[PolicyDocument]:
    """Return PDF documents located under ``policies_dir``."""

    pdf_paths = sorted(policies_dir.glob("*.pdf"))
    documents: List[PolicyDocument] = []
    for path in pdf_paths:
        document_id = re.sub(r"[^a-z0-9]+", "_", path.stem.lower()).strip("_")
        title = path.name
        documents.append(PolicyDocument(document_id=document_id, title=title, path=path))
    return documents


def load_policy_chunks(
    documents: Sequence[PolicyDocument], *, chunk_size: int = 220, overlap: int = 40
) -> List[PolicyChunk]:
    """Extract cleaned chunks from each provided document."""

    chunks: List[PolicyChunk] = []
    for doc in documents:
        try:
            doc_chunks = _extract_chunks_from_pdf(doc, chunk_size=chunk_size, overlap=overlap)
        except Exception as exc:
            logger.error("Failed to parse %s: %s", doc.path, exc)
            continue
        chunks.extend(doc_chunks)
    return chunks


def _extract_chunks_from_pdf(
    document: PolicyDocument, *, chunk_size: int, overlap: int
) -> List[PolicyChunk]:
    pages: List[dict[str, Any]] = []
    with pdfplumber.open(str(document.path)) as pdf:
        for index, page in enumerate(pdf.pages):
            lines = _extract_page_lines(page)
            pages.append({"page_number": index + 1, "lines": lines})

    if not pages:
        return []

    header_lines, footer_lines = _detect_repeating_headers_and_footers(pages)
    cleaned_pages = []
    for page in pages:
        cleaned_lines = _clean_page_lines(
            page["lines"],
            headers=header_lines,
            footers=footer_lines,
        )
        normalised_text = _normalise_text("\n".join(cleaned_lines))
        cleaned_pages.append({"page_number": page["page_number"], "text": normalised_text})

    chunks: List[PolicyChunk] = []
    for page in cleaned_pages:
        page_text = page["text"].strip()
        if not page_text:
            continue
        page_chunks = list(
            _chunk_text(page_text, chunk_size=chunk_size, overlap=overlap)
        ) or [page_text]
        for chunk_idx, chunk_text in enumerate(page_chunks):
            chunk_id = f"{document.document_id}_p{page['page_number']}_c{chunk_idx}"
            chunks.append(
                PolicyChunk(
                    chunk_id=chunk_id,
                    document_id=document.document_id,
                    title=document.title,
                    page_number=page["page_number"],
                    chunk_index=chunk_idx,
                    text=chunk_text,
                )
            )
    return chunks


def _extract_page_lines(page: Any) -> List[str]:
    text_segments: List[str] = []
    try:
        body_text = page.extract_text() or ""
    except Exception:  # pragma: no cover - pdfplumber may fail on complex pages
        body_text = ""
    if body_text:
        text_segments.append(body_text)

    try:
        tables = page.extract_tables() or []
    except Exception:  # pragma: no cover - table extraction best effort
        tables = []
    for table in tables:
        rows: List[str] = []
        for row in table:
            cells = [cell.strip() if isinstance(cell, str) else "" for cell in row]
            if any(cell for cell in cells):
                rows.append(" | ".join(cells))
        if rows:
            text_segments.append("\n".join(rows))

    combined_text = "\n".join(segment.strip() for segment in text_segments if segment.strip())
    return [line.strip() for line in combined_text.splitlines() if line.strip()]


def _detect_repeating_headers_and_footers(
    pages: Sequence[dict[str, Sequence[str]]], threshold: float = 0.6
) -> tuple[set[str], set[str]]:
    header_counter: Counter[str] = Counter()
    footer_counter: Counter[str] = Counter()
    for page in pages:
        lines = page["lines"]
        header_counter.update(lines[:3])
        footer_counter.update(lines[-3:])
    total_pages = max(len(pages), 1)
    header_lines = {line for line, count in header_counter.items() if count / total_pages >= threshold}
    footer_lines = {line for line, count in footer_counter.items() if count / total_pages >= threshold}
    return header_lines, footer_lines


def _clean_page_lines(
    lines: Sequence[str], *, headers: Iterable[str], footers: Iterable[str]
) -> List[str]:
    header_set = set(headers)
    footer_set = set(footers)
    cleaned: List[str] = []
    page_number_pattern = re.compile(r"^(page\s*)?\d{1,4}[a-zA-Z]*$", re.IGNORECASE)
    divider_pattern = re.compile(r"^[-_–—•\s]*$")
    for line in lines:
        if line in header_set or line in footer_set:
            continue
        if page_number_pattern.match(line):
            continue
        if divider_pattern.match(line):
            continue
        cleaned.append(line)
    return cleaned


def _chunk_text(text: str, *, chunk_size: int, overlap: int) -> Iterable[str]:
    sentences = _split_into_sentences(text)
    if not sentences:
        return []

    sentence_word_counts = [len(sentence.split()) or 1 for sentence in sentences]
    total_sentences = len(sentences)
    start_index = 0

    while start_index < total_sentences:
        previous_start = start_index
        current_words = 0
        chunk_sentences: List[str] = []
        end_index = start_index

        while end_index < total_sentences:
            sentence = sentences[end_index]
            sentence_words = sentence_word_counts[end_index]
            if chunk_sentences and current_words + sentence_words > chunk_size:
                break
            chunk_sentences.append(sentence)
            current_words += sentence_words
            end_index += 1

        if not chunk_sentences:
            chunk_sentences.append(sentences[end_index])
            end_index += 1

        yield " ".join(chunk_sentences)

        if end_index >= total_sentences:
            break

        overlap_words_remaining = overlap
        next_start = end_index
        while next_start > previous_start and overlap_words_remaining > 0:
            next_start -= 1
            overlap_words_remaining -= sentence_word_counts[next_start]
        start_index = max(next_start, previous_start + 1)
        if start_index >= end_index:
            start_index = end_index - 1 if end_index - 1 > previous_start else end_index


def _split_into_sentences(text: str) -> List[str]:
    sentence_end_pattern = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
    segments = sentence_end_pattern.split(text)
    sentences: List[str] = []
    for segment in segments:
        cleaned = _normalise_text(segment)
        if cleaned:
            sentences.append(cleaned)
    return sentences


def _normalise_text(text: str) -> str:
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    return re.sub(r"\s+", " ", text).strip()
