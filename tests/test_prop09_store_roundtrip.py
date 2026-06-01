"""Property 9: Round-trip de armazenamento no Vector Store."""

from hypothesis import given, settings
from hypothesis.strategies import text, lists, characters

from core.models import Chunk
from tests.fakes.hash_embeddings import HashEmbeddings
from tests.fakes.in_memory_store import InMemoryVectorStore

_safe_text = text(alphabet=characters(blacklist_categories=("Cs",)), min_size=1, max_size=100)


@settings(max_examples=100)
@given(texts=lists(_safe_text, min_size=1, max_size=10))
def test_store_roundtrip(texts):
    emb = HashEmbeddings(dimension=32)
    store = InMemoryVectorStore()

    chunks = [
        Chunk(
            chunk_id=f"c{i}",
            text=t,
            document_name="doc.pdf",
            page_number=1,
            index=i,
            char_count=len(t),
        )
        for i, t in enumerate(texts)
    ]
    vectors = emb.embed_passages(texts).embeddings

    store.upsert_document("doc.pdf", chunks, vectors)

    # All chunks retrievable
    stored = store.all_chunks()
    assert len(stored) == len(chunks)
    stored_ids = {c.chunk_id for c in stored}
    for c in chunks:
        assert c.chunk_id in stored_ids

    # Count matches
    assert store.count() == len(chunks)
