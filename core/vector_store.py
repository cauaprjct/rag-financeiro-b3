"""Vector store: Protocol and ChromaVectorStore implementation."""

from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

from core.models import Chunk, DeleteResult, DocumentInfo, ScoredChunk

logger = logging.getLogger(__name__)


@runtime_checkable
class VectorStore(Protocol):
    @property
    def embedding_model_id(self) -> str: ...

    def upsert_document(
        self, document_name: str, chunks: list[Chunk], embeddings: list[list[float]]
    ) -> None: ...

    def query_dense(
        self, query_embedding: list[float], top_k: int, document_filter: list[str] | None = None
    ) -> list[ScoredChunk]: ...

    def all_chunks(self) -> list[Chunk]: ...

    def list_documents(self) -> list[DocumentInfo]: ...

    def delete_document(self, document_name: str) -> DeleteResult: ...

    def clear(self) -> None: ...

    def count(self) -> int: ...


class ChromaVectorStore:
    """Chroma-backed persistent vector store."""

    def __init__(self, persist_dir: str, collection_name: str, embedding_model_id: str) -> None:
        import chromadb

        self._embedding_model_id = embedding_model_id
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"embedding_model_id": embedding_model_id},
        )

    @property
    def embedding_model_id(self) -> str:
        return self._embedding_model_id

    def upsert_document(
        self, document_name: str, chunks: list[Chunk], embeddings: list[list[float]]
    ) -> None:
        # Remove old chunks for this document first
        self._delete_by_name(document_name)
        if not chunks:
            return
        ids = [c.chunk_id for c in chunks]
        documents = [c.text for c in chunks]
        metadatas = [
            {
                "document_name": c.document_name,
                "page_number": c.page_number,
                "chunk_id": c.chunk_id,
                "index": c.index,
                "char_count": c.char_count,
            }
            for c in chunks
        ]
        self._collection.upsert(
            ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas
        )

    def query_dense(
        self, query_embedding: list[float], top_k: int, document_filter: list[str] | None = None
    ) -> list[ScoredChunk]:
        where = {"document_name": {"$in": document_filter}} if document_filter else None
        try:
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, self._collection.count()),
                where=where,
            )
        except Exception:
            return []
        if not results["ids"] or not results["ids"][0]:
            return []

        scored: list[ScoredChunk] = []
        for i, cid in enumerate(results["ids"][0]):
            meta = results["metadatas"][0][i]
            text = results["documents"][0][i]
            dist = results["distances"][0][i] if results["distances"] else 0.0
            score = 1.0 / (1.0 + dist)
            chunk = Chunk(
                chunk_id=cid,
                text=text,
                document_name=meta["document_name"],
                page_number=meta["page_number"],
                index=meta["index"],
                char_count=meta["char_count"],
            )
            scored.append(ScoredChunk(chunk=chunk, score=score))
        return scored

    def all_chunks(self) -> list[Chunk]:
        if self._collection.count() == 0:
            return []
        results = self._collection.get()
        chunks: list[Chunk] = []
        for i, cid in enumerate(results["ids"]):
            meta = results["metadatas"][i]
            chunks.append(
                Chunk(
                    chunk_id=cid,
                    text=results["documents"][i],
                    document_name=meta["document_name"],
                    page_number=meta["page_number"],
                    index=meta["index"],
                    char_count=meta["char_count"],
                )
            )
        return chunks

    def list_documents(self) -> list[DocumentInfo]:
        if self._collection.count() == 0:
            return []
        results = self._collection.get()
        doc_counts: dict[str, int] = {}
        for meta in results["metadatas"]:
            name = meta["document_name"]
            doc_counts[name] = doc_counts.get(name, 0) + 1
        return [DocumentInfo(name=n, chunk_count=c) for n, c in doc_counts.items()]

    def delete_document(self, document_name: str) -> DeleteResult:
        count_before = self._collection.count()
        self._delete_by_name(document_name)
        count_after = self._collection.count()
        deleted = count_before - count_after
        msg = "" if deleted > 0 else f"Documento '{document_name}' não encontrado"
        return DeleteResult(document_name=document_name, deleted_count=deleted, message=msg)

    def clear(self) -> None:
        if self._collection.count() > 0:
            all_ids = self._collection.get()["ids"]
            self._collection.delete(ids=all_ids)

    def count(self) -> int:
        return self._collection.count()

    def _delete_by_name(self, document_name: str) -> None:
        try:
            self._collection.delete(where={"document_name": document_name})
        except Exception:
            pass
