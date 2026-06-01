"""Embedding module: Protocol and LocalE5Embeddings implementation."""

from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

from core.models import EmbedBatchResult

logger = logging.getLogger(__name__)


@runtime_checkable
class EmbeddingModule(Protocol):
    @property
    def model_id(self) -> str: ...

    @property
    def dimension(self) -> int: ...

    def embed_passages(self, texts: list[str]) -> EmbedBatchResult: ...

    def embed_query(self, text: str) -> list[float] | None: ...


class LocalE5Embeddings:
    """Sentence-transformers based embeddings with e5 prefix convention."""

    def __init__(self, model_id: str, device: str = "auto") -> None:
        import torch
        from sentence_transformers import SentenceTransformer

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self._model_id = model_id
        self._model = SentenceTransformer(model_id, device=device)
        self._dimension = self._model.get_sentence_embedding_dimension()
        logger.info(f"Loaded embedding model: {model_id} (dim={self._dimension}, device={device})")

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_passages(self, texts: list[str]) -> EmbedBatchResult:
        result = EmbedBatchResult()
        for text in texts:
            try:
                prefixed = f"passage: {text}"
                vec = self._model.encode(prefixed, normalize_embeddings=True).tolist()
                result.embeddings.append(vec)
                result.errors.append("")
            except Exception as e:
                result.embeddings.append(None)
                result.errors.append(str(e))
                logger.warning(f"Embedding failed for passage: {e}")
        return result

    def embed_query(self, text: str) -> list[float] | None:
        try:
            prefixed = f"query: {text}"
            return self._model.encode(prefixed, normalize_embeddings=True).tolist()
        except Exception as e:
            logger.warning(f"Embedding failed for query: {e}")
            return None
