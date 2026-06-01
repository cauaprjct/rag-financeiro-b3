"""Property 4: IDs de chunk determinísticos e únicos."""

from hypothesis import given, settings, assume
from hypothesis.strategies import text, integers

from core.chunking import chunk_document
from core.models import PageText


@settings(max_examples=100)
@given(
    content=text(min_size=5, max_size=1000),
    chunk_size=integers(min_value=3, max_value=200),
    overlap=integers(min_value=0, max_value=199),
)
def test_chunk_ids_deterministic_and_unique(content, chunk_size, overlap):
    assume(overlap < chunk_size)
    assume(content.strip())
    pages = [PageText(page_number=1, text=content)]

    chunks1 = chunk_document(pages, "doc.pdf", chunk_size, overlap)
    chunks2 = chunk_document(pages, "doc.pdf", chunk_size, overlap)

    # Deterministic
    ids1 = [c.chunk_id for c in chunks1]
    ids2 = [c.chunk_id for c in chunks2]
    assert ids1 == ids2

    # Unique
    assert len(set(ids1)) == len(ids1)
