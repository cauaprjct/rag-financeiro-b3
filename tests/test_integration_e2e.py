"""Teste de integração e2e offline: indexação → retrieval → geração."""

from pathlib import Path

from core.config import AppConfig
from core.generation import GenerationModule
from core.pipeline import Pipeline
from core.retrieval import RetrievalModule
from tests.fakes.fake_llm import FakeLLM
from tests.fakes.hash_embeddings import HashEmbeddings
from tests.fakes.in_memory_store import InMemoryVectorStore

_FIXTURES = Path(__file__).parent / "fixtures"


def _build_offline_pipeline(llm_response="A receita líquida foi de R$ 1 bilhão. [1]"):
    cfg = AppConfig(
        groq_api_key="fake",
        gemini_api_key="",
        chunk_size=128,
        chunk_overlap=16,
        top_k=3,
        embedding_model_id="fake/hash-embeddings",
        rerank_enabled=False,
        reranker_model_id="",
        rerank_candidates_n=20,
        device="cpu",
    )
    emb = HashEmbeddings(dimension=64)
    store = InMemoryVectorStore()
    llm = FakeLLM(response=llm_response)
    return (
        Pipeline(
            cfg=cfg,
            embeddings=emb,
            store=store,
            retrieval=RetrievalModule(store, emb),
            generation=GenerationModule(llm),
        ),
        llm,
    )


def test_e2e_index_retrieve_generate():
    pipeline, llm = _build_offline_pipeline()

    # Index real PDF fixtures
    results = pipeline.index_files(
        [
            _FIXTURES / "valid.pdf",
            _FIXTURES / "multipage.pdf",
        ]
    )
    assert all(r.success for r in results)
    assert pipeline.store.count() > 0

    # Query
    answer = pipeline.query("Qual foi a receita líquida?")

    # Generated answer with coherent citations
    assert answer.status == "ok"
    assert answer.answer == "A receita líquida foi de R$ 1 bilhão. [1]"
    assert len(answer.citations) > 0
    for cit in answer.citations:
        assert cit.document_name in {"valid.pdf", "multipage.pdf"}
        assert cit.page_number >= 1
        assert cit.snippet

    # LLM was invoked with a prompt containing the question
    assert len(llm.calls) == 1
    assert "receita líquida" in llm.calls[0].lower()


def test_e2e_empty_collection_no_llm_call():
    pipeline, llm = _build_offline_pipeline()
    # No indexing
    answer = pipeline.query("Pergunta sem documentos")
    assert answer.status == "no_context"
    assert llm.calls == []


def test_e2e_document_filter():
    pipeline, llm = _build_offline_pipeline()
    pipeline.index_files([_FIXTURES / "valid.pdf", _FIXTURES / "multipage.pdf"])

    answer = pipeline.query("receita", document_filter=["valid.pdf"])
    for cit in answer.citations:
        assert cit.document_name == "valid.pdf"
