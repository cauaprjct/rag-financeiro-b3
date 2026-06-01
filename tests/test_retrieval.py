"""Testes de exemplo do Retrieval_Module."""

from core.retrieval import RetrievalModule, RetrievalParams
from tests.fakes.hash_embeddings import HashEmbeddings
from tests.fakes.in_memory_store import InMemoryVectorStore


def test_empty_collection_no_search():
    """Req 5.4: coleção vazia retorna lista vazia + mensagem, sem buscar."""
    emb = HashEmbeddings(dimension=32)
    store = InMemoryVectorStore()
    module = RetrievalModule(store, emb)

    result = module.retrieve(RetrievalParams(query="receita líquida"))

    assert result.scored_chunks == []
    assert "nenhum documento" in result.message.lower()
