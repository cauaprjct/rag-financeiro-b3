"""In-memory vector store for offline testing. Same interface as VectorStore."""

from __future__ import annotations

import math

from core.models import Chunk, DeleteResult, DocumentInfo, ScoredChunk


class InMemoryVectorStore:
    """In-memory vector store with cosine similarity for dense search."""

    def __init__(self, embedding_model_id: str = "fake/hash-embeddings") -> None:
        self._embedding_model_id = embedding_model_id
        self._chunks: list[Chunk] = []
        self._embeddings: list[list[float]] = []

    @property
    def embedding_model_id(self) -> str:
        return self._embedding_model_id

    def upsert_document(
        self, document_name: str, chunks: list[Chunk], embeddings: list[list[float]]
    ) -> None:
        # Remove old chunks for this document
        self._remove_by_name(document_name)
        for chunk, emb in zip(chunks, embeddings):
            self._chunks.append(chunk)
            self._embeddings.append(emb)

    def query_dense(
        self, query_embedding: list[float], top_k: int, document_filter: list[str] | None = None
    ) -> list[ScoredChunk]:
        scored: list[ScoredChunk] = []
        for chunk, emb in zip(self._chunks, self._embeddings):
            if document_filter and chunk.document_name not in document_filter:
                continue
            score = _cosine_similarity(query_embedding, emb)
            scored.append(ScoredChunk(chunk=chunk, score=score))
        scored.sort(key=lambda s: s.score, reverse=True)
        return scored[:top_k]

    def all_chunks(self) -> list[Chunk]:
        return list(self._chunks)

    def list_documents(self) -> list[DocumentInfo]:
        doc_counts: dict[str, int] = {}
        for c in self._chunks:
            doc_counts[c.document_name] = doc_counts.get(c.document_name, 0) + 1
        return [DocumentInfo(name=n, chunk_count=ct) for n, ct in doc_counts.items()]

    def delete_document(self, document_name: str) -> DeleteResult:
        before = len(self._chunks)
        self._remove_by_name(document_name)
        deleted = before - len(self._chunks)
        msg = "" if deleted > 0 else f"Documento '{document_name}' não encontrado"
        return DeleteResult(document_name=document_name, deleted_count=deleted, message=msg)

    def clear(self) -> None:
        self._chunks.clear()
        self._embeddings.clear()

    def count(self) -> int:
        return len(self._chunks)

    def _remove_by_name(self, document_name: str) -> None:
        pairs = [
            (c, e)
            for c, e in zip(self._chunks, self._embeddings)
            if c.document_name != document_name
        ]
        if pairs:
            self._chunks, self._embeddings = [list(x) for x in zip(*pairs)]
        else:
            self._chunks, self._embeddings = [], []


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
