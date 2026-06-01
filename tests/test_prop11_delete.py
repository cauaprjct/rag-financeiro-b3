"""Property 11: Remoção de documento elimina todos os seus chunks."""

from hypothesis import given, settings
from hypothesis.strategies import text, lists, characters

from core.models import Chunk
from tests.fakes.hash_embeddings import HashEmbeddings
from tests.fakes.in_memory_store import InMemoryVectorStore

_safe_text = text(alphabet=characters(blacklist_categories=("Cs",)), min_size=1, max_size=100)


@settings(max_examples=100)
@given(
    texts_a=lists(_safe_text, min_size=1, max_size=5),
    texts_b=lists(_safe_text, min_size=1, max_size=5),
)
def test_delete_removes_all_chunks(texts_a, texts_b):
    emb = HashEmbeddings(dimension=32)
    store = InMemoryVectorStore()

    chunks_a = [
        Chunk(
            chunk_id=f"a{i}",
            text=t,
            document_name="a.pdf",
            page_number=1,
            index=i,
            char_count=len(t),
        )
        for i, t in enumerate(texts_a)
    ]
    chunks_b = [
        Chunk(
            chunk_id=f"b{i}",
            text=t,
            document_name="b.pdf",
            page_number=1,
            index=i,
            char_count=len(t),
        )
        for i, t in enumerate(texts_b)
    ]

    store.upsert_document("a.pdf", chunks_a, emb.embed_passages(texts_a).embeddings)
    store.upsert_document("b.pdf", chunks_b, emb.embed_passages(texts_b).embeddings)

    result = store.delete_document("a.pdf")
    assert result.deleted_count == len(texts_a)

    # No chunks from a.pdf remain
    remaining = store.all_chunks()
    for c in remaining:
        assert c.document_name != "a.pdf"

    # b.pdf chunks intact
    assert store.count() == len(texts_b)
