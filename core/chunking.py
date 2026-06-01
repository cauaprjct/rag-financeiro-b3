"""Chunking module: sliding window over tokens with tiktoken."""

from __future__ import annotations

import hashlib
import math
import warnings

import tiktoken

from core.models import Chunk, PageText

_ENC = tiktoken.get_encoding("cl100k_base")


def make_chunk_id(document_name: str, index: int, text: str) -> str:
    """Deterministic SHA-1 chunk id."""
    payload = f"{document_name}:{index}:{text}"
    return hashlib.sha1(payload.encode()).hexdigest()


def normalize_params(chunk_size: int, overlap: int) -> int:
    """Return adjusted overlap. Warns if overlap >= chunk_size."""
    if overlap >= chunk_size:
        adjusted = math.floor(0.10 * chunk_size)
        warnings.warn(f"overlap ({overlap}) >= chunk_size ({chunk_size}), ajustado para {adjusted}")
        return adjusted
    return overlap


def chunk_document(
    pages: list[PageText],
    document_name: str,
    chunk_size: int,
    overlap: int,
) -> list[Chunk]:
    """Split pages into token-based chunks using a sliding window."""
    overlap = normalize_params(chunk_size, overlap)
    full_text = "\n".join(p.text for p in pages)
    if not full_text.strip():
        return []

    tokens = _ENC.encode(full_text)
    if not tokens:
        return []

    chunks: list[Chunk] = []
    step = chunk_size - overlap
    i = 0
    idx = 0
    while i < len(tokens):
        window = tokens[i : i + chunk_size]
        text = _ENC.decode(window)
        page_number = _find_page(pages, i, tokens)
        chunk_id = make_chunk_id(document_name, idx, text)
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                text=text,
                document_name=document_name,
                page_number=page_number,
                index=idx,
                char_count=len(text),
            )
        )
        idx += 1
        i += step
    return chunks


def _find_page(pages: list[PageText], token_offset: int, all_tokens: list[int]) -> int:
    """Find which page a token offset belongs to."""
    acc = 0
    for p in pages:
        page_tokens = len(_ENC.encode(p.text))
        sep = 1 if acc > 0 else 0  # newline separator
        if token_offset < acc + page_tokens + sep:
            return p.page_number
        acc += page_tokens + sep
    return pages[-1].page_number if pages else 1
