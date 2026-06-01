"""Fake cross-encoder reranker. Deterministic hash-based scoring, no model download."""

from __future__ import annotations

import hashlib
import math
import struct

from core.models import ScoredChunk


class FakeCrossEncoder:
    """Deterministic reranker based on hash of (question, text) pair."""

    def rerank(self, question: str, candidates: list[ScoredChunk], top_k: int) -> list[ScoredChunk]:
        if not candidates:
            return []

        scored = []
        for sc in candidates:
            raw = self._hash_score(question, sc.chunk.text)
            norm_score = 1.0 / (1.0 + math.exp(-raw))
            scored.append(ScoredChunk(chunk=sc.chunk, score=norm_score))

        scored.sort(key=lambda s: s.score, reverse=True)
        return scored[:top_k]

    def _hash_score(self, question: str, text: str) -> float:
        """Deterministic score in [-3, 3] from hash of pair."""
        payload = f"{question}||{text}".encode()
        h = hashlib.sha256(payload).digest()
        val = struct.unpack(">I", h[:4])[0]
        return (val / 0xFFFFFFFF) * 6.0 - 3.0  # map to [-3, 3]
