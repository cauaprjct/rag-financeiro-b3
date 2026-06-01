"""Property 10: Upsert por documento é idempotente (sem duplicatas)."""

from hypothesis import given, settings
from hypothesis.strategies import text, lists, characters, integers

from core.models import Chunk
from tests.fakes.hash_embeddings import HashEmbeddings
from tests.fakes.in_memory_store import InMemoryVectorStore

_safe_text = text(alphabet=characters(blacklist_categories=("Cs",)), min_size=1, max_size=100)


@settings(max_examples=100)
@given(
    texts=lists(_safe_text, min_size=1, max_size=8),
    repeats=integers(min_value=2, max_value=5),
)
def test_upsert_idempotent(texts, repeats):
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

    for _ in range(repeats):
        store.upsert_document("doc.pdf", chunks, vectors)

    # No duplicates
    assert store.count() == len(chunks)
