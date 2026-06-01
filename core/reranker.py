"""Reranker module: Protocol, NoOpReranker, CrossEncoderReranker."""

from __future__ import annotations

import logging
import math
from typing import Protocol, runtime_checkable

from core.models import ScoredChunk

logger = logging.getLogger(__name__)


@runtime_checkable
class RerankerModule(Protocol):
    def rerank(
        self, question: str, candidates: list[ScoredChunk], top_k: int
    ) -> list[ScoredChunk]: ...


class NoOpReranker:
    """Returns candidates truncated to top_k without reranking. Default/fallback."""

    def rerank(self, question: str, candidates: list[ScoredChunk], top_k: int) -> list[ScoredChunk]:
        return candidates[:top_k]


class CrossEncoderReranker:
    """Cross-encoder reranker using sentence-transformers. Lazy loading."""

    def __init__(self, model_id: str = "BAAI/bge-reranker-v2-m3", device: str = "auto") -> None:
        self._model_id = model_id
        self._device = device
        self._model = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from sentence_transformers import CrossEncoder

        device = self._device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self._model = CrossEncoder(self._model_id, device=device)
        logger.info(f"Loaded reranker: {self._model_id} (device={device})")

    def rerank(self, question: str, candidates: list[ScoredChunk], top_k: int) -> list[ScoredChunk]:
        self._load()
        if not candidates:
            return []

        pairs = [(question, sc.chunk.text) for sc in candidates]
        raw_scores = self._model.predict(pairs)

        # Normalize with sigmoid to [0, 1]
        scored = []
        for sc, raw in zip(candidates, raw_scores):
            norm_score = 1.0 / (1.0 + math.exp(-float(raw)))
            scored.append(ScoredChunk(chunk=sc.chunk, score=norm_score))

        scored.sort(key=lambda s: s.score, reverse=True)
        return scored[:top_k]
