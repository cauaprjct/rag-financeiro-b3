"""Pipeline factory: wires config, embeddings, store, retrieval, generation, evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from core.chunking import chunk_document
from core.config import AppConfig, load_config
from core.generation import GenerationModule, GenerationResult
from core.ingestion import ingest_batch
from core.models import IngestionResult
from core.retrieval import RetrievalModule, RetrievalParams


@dataclass
class Pipeline:
    cfg: AppConfig
    embeddings: object
    store: object
    retrieval: RetrievalModule
    generation: GenerationModule
    reranker: object | None = None

    def index_files(self, paths: list[Path]) -> list[IngestionResult]:
        """Ingest, chunk, embed and upsert each file."""
        results = ingest_batch(paths)
        for res in results:
            if not res.success or not res.pages:
                continue
            chunks = chunk_document(
                res.pages, res.document_name, self.cfg.chunk_size, self.cfg.chunk_overlap
            )
            if not chunks:
                continue
            batch = self.embeddings.embed_passages([c.text for c in chunks])
            valid = [(c, e) for c, e in zip(chunks, batch.embeddings) if e is not None]
            if valid:
                vc, ve = zip(*valid)
                self.store.upsert_document(res.document_name, list(vc), list(ve))
        return results

    def query(self, question: str, *, document_filter: list[str] | None = None) -> GenerationResult:
        """Retrieve relevant chunks and generate a grounded answer."""
        params = RetrievalParams(
            query=question,
            top_k=self.cfg.top_k,
            document_filter=document_filter,
            reranker=self.reranker,
            rerank_enabled=self.cfg.rerank_enabled and self.reranker is not None,
            rerank_candidates_n=self.cfg.rerank_candidates_n,
        )
        result = self.retrieval.retrieve(params)
        return self.generation.generate(question, result.scored_chunks)


def build_pipeline(
    cfg: AppConfig | None = None,
    *,
    secrets: dict | None = None,
    embeddings=None,
    store=None,
    llm=None,
) -> Pipeline:
    """Build a Pipeline. Components can be injected for offline testing."""
    if cfg is None:
        cfg = load_config(secrets)

    if embeddings is None:
        from core.embeddings import LocalE5Embeddings

        embeddings = LocalE5Embeddings(cfg.embedding_model_id, device=cfg.device)

    if store is None:
        from core.vector_store import ChromaVectorStore

        store = ChromaVectorStore(
            persist_dir="data/chroma",
            collection_name="b3_documents",
            embedding_model_id=cfg.embedding_model_id,
        )

    if llm is None:
        from core.llm_provider import make_llm_provider

        llm = make_llm_provider(cfg)

    reranker = None
    if cfg.rerank_enabled:
        from core.reranker import CrossEncoderReranker

        reranker = CrossEncoderReranker(cfg.reranker_model_id, device=cfg.device)

    return Pipeline(
        cfg=cfg,
        embeddings=embeddings,
        store=store,
        retrieval=RetrievalModule(store, embeddings),
        generation=GenerationModule(llm),
        reranker=reranker,
    )
