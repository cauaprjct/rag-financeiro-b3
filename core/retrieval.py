"""Retrieval module: dense search, BM25 hybrid, RRF fusion, reranking, filters."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from core.models import Chunk, RetrievalResult, ScoredChunk

logger = logging.getLogger(__name__)


@dataclass
class RetrievalParams:
    query: str = ""
    top_k: int = 5
    hybrid: bool = True
    document_filter: list[str] | None = None
    min_relevance_score: float = 0.0
    reranker: object | None = None
    rerank_enabled: bool = False
    rerank_candidates_n: int = 20


def reciprocal_rank_fusion(
    ranked_lists: list[list[ScoredChunk]], k: int = 60, top_k: int = 5
) -> list[ScoredChunk]:
    """Fuse multiple ranked lists using RRF. No duplicates, ordered desc, limited to top_k."""
    scores: dict[str, float] = {}
    chunk_map: dict[str, Chunk] = {}

    for ranked in ranked_lists:
        for rank, sc in enumerate(ranked):
            cid = sc.chunk.chunk_id
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
            chunk_map[cid] = sc.chunk

    # Sort by RRF score descending
    sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    # Normalize to [0, 1]
    max_score = sorted_items[0][1] if sorted_items else 1.0
    result: list[ScoredChunk] = []
    for cid, score in sorted_items[:top_k]:
        normalized = score / max_score if max_score > 0 else 0.0
        result.append(ScoredChunk(chunk=chunk_map[cid], score=normalized))
    return result


def _normalize_scores(scored: list[ScoredChunk]) -> list[ScoredChunk]:
    """Normalize scores to [0, 1] range."""
    if not scored:
        return []
    max_s = max(sc.score for sc in scored)
    min_s = min(sc.score for sc in scored)
    rng = max_s - min_s
    if rng == 0:
        return [ScoredChunk(chunk=sc.chunk, score=1.0) for sc in scored]
    return [ScoredChunk(chunk=sc.chunk, score=(sc.score - min_s) / rng) for sc in scored]


class RetrievalModule:
    """Retrieval with dense search, optional BM25 hybrid via RRF."""

    def __init__(self, vector_store, embedding_module) -> None:
        self._store = vector_store
        self._emb = embedding_module

    def retrieve(self, params: RetrievalParams) -> RetrievalResult:
        # Empty collection check
        if self._store.count() == 0:
            return RetrievalResult(scored_chunks=[], message="Nenhum documento carregado")

        query_vec = self._emb.embed_query(params.query)
        if query_vec is None:
            return RetrievalResult(scored_chunks=[], message="Falha ao gerar embedding da consulta")

        # Determine how many candidates to fetch
        if params.rerank_enabled and params.reranker:
            n_candidates = max(params.rerank_candidates_n, params.top_k)
        else:
            n_candidates = params.top_k * 4 if params.hybrid else params.top_k

        # Dense search
        dense_results = self._store.query_dense(
            query_vec, top_k=n_candidates, document_filter=params.document_filter
        )

        if params.hybrid:
            # BM25 search
            all_chunks = self._store.all_chunks()
            if params.document_filter:
                all_chunks = [c for c in all_chunks if c.document_name in params.document_filter]
            bm25_results = self._bm25_search(params.query, all_chunks, n_candidates)
            # RRF fusion - get more candidates if reranking
            rrf_top = n_candidates if (params.rerank_enabled and params.reranker) else params.top_k
            scored = reciprocal_rank_fusion([dense_results, bm25_results], top_k=rrf_top)
        else:
            scored = _normalize_scores(dense_results)[:n_candidates]

        # Reranking step
        if params.rerank_enabled and params.reranker:
            try:
                scored = params.reranker.rerank(params.query, scored, params.top_k)
            except Exception as e:
                logger.warning(f"Reranker failed, falling back to base retrieval: {e}")
                scored = scored[: params.top_k]
        else:
            scored = scored[: params.top_k]

        # Apply min_relevance_score filter
        if params.min_relevance_score > 0:
            scored = [sc for sc in scored if sc.score >= params.min_relevance_score]

        return RetrievalResult(scored_chunks=scored)

    def _bm25_search(self, query: str, chunks: list[Chunk], top_k: int) -> list[ScoredChunk]:
        if not chunks:
            return []
        corpus = [c.text.split() for c in chunks]
        bm25 = BM25Okapi(corpus)
        tokenized_query = query.split()
        scores = bm25.get_scores(tokenized_query)

        indexed = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
        return [ScoredChunk(chunk=chunks[i], score=s) for i, s in indexed if s > 0]
