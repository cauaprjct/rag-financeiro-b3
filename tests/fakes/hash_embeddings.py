"""Fake deterministic embeddings based on text hash. No network, no model."""

from __future__ import annotations

import hashlib
import struct

from core.models import EmbedBatchResult


class HashEmbeddings:
    """Deterministic embeddings derived from text hash. Fixed dimension, batch-invariant."""

    def __init__(self, dimension: int = 64) -> None:
        self._dimension = dimension
        self._model_id = "fake/hash-embeddings"

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
                vec = self._hash_to_vector(f"passage: {text}")
                result.embeddings.append(vec)
                result.errors.append("")
            except Exception as e:
                result.embeddings.append(None)
                result.errors.append(str(e))
        return result

    def embed_query(self, text: str) -> list[float] | None:
        return self._hash_to_vector(f"query: {text}")

    def _hash_to_vector(self, text: str) -> list[float]:
        # Generate deterministic vector by repeatedly hashing
        vec: list[float] = []
        seed = text.encode()
        while len(vec) < self._dimension:
            h = hashlib.sha256(seed + struct.pack(">I", len(vec))).digest()
            # Each sha256 gives 32 bytes = 8 floats (4 bytes each as normalized)
            for i in range(0, 32, 4):
                if len(vec) >= self._dimension:
                    break
                val = struct.unpack(">I", h[i : i + 4])[0]
                vec.append((val / 0xFFFFFFFF) * 2 - 1)  # normalize to [-1, 1]
        return vec
