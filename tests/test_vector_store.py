"""Testes de exemplo do Vector_Store."""

from core.models import Chunk
from core.vector_store import VectorStore
from tests.fakes.hash_embeddings import HashEmbeddings
from tests.fakes.in_memory_store import InMemoryVectorStore


def test_in_memory_satisfies_protocol():
    store = InMemoryVectorStore()
    assert isinstance(store, VectorStore)


def test_delete_nonexistent_document_returns_message():
    """Req 4.9: remover documento inexistente retorna mensagem."""
    store = InMemoryVectorStore()
    result = store.delete_document("ghost.pdf")
    assert result.deleted_count == 0
    assert "não encontrado" in result.message


def test_query_dense_returns_scored_chunks():
    """Basic dense query returns scored results."""
    emb = HashEmbeddings(dimension=32)
    store = InMemoryVectorStore()

    chunks = [
        Chunk(
            chunk_id="c0",
            text="receita líquida",
            document_name="doc.pdf",
            page_number=1,
            index=0,
            char_count=15,
        ),
        Chunk(
            chunk_id="c1",
            text="despesa operacional",
            document_name="doc.pdf",
            page_number=2,
            index=1,
            char_count=19,
        ),
    ]
    vectors = emb.embed_passages(["receita líquida", "despesa operacional"]).embeddings
    store.upsert_document("doc.pdf", chunks, vectors)

    q_vec = emb.embed_query("receita")
    results = store.query_dense(q_vec, top_k=2)
    assert len(results) == 2
    assert all(r.score is not None for r in results)


def test_document_filter():
    """Req 4.6: query with document filter."""
    emb = HashEmbeddings(dimension=32)
    store = InMemoryVectorStore()

    c1 = Chunk(
        chunk_id="a0", text="texto a", document_name="a.pdf", page_number=1, index=0, char_count=7
    )
    c2 = Chunk(
        chunk_id="b0", text="texto b", document_name="b.pdf", page_number=1, index=0, char_count=7
    )
    store.upsert_document("a.pdf", [c1], emb.embed_passages(["texto a"]).embeddings)
    store.upsert_document("b.pdf", [c2], emb.embed_passages(["texto b"]).embeddings)

    q_vec = emb.embed_query("texto")
    results = store.query_dense(q_vec, top_k=10, document_filter=["a.pdf"])
    assert all(r.chunk.document_name == "a.pdf" for r in results)
