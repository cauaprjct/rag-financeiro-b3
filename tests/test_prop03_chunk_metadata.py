"""Property 3: Metadados de origem em todo chunk."""

from hypothesis import given, settings, assume
from hypothesis.strategies import text, integers

from core.chunking import chunk_document
from core.models import PageText


@settings(max_examples=100)
@given(
    content=text(min_size=1, max_size=1000),
    chunk_size=integers(min_value=1, max_value=200),
    overlap=integers(min_value=0, max_value=199),
)
def test_all_chunks_have_metadata(content, chunk_size, overlap):
    assume(overlap < chunk_size)
    assume(content.strip())
    doc_name = "report.pdf"
    pages = [PageText(page_number=1, text=content)]
    chunks = chunk_document(pages, doc_name, chunk_size, overlap)
    for c in chunks:
        assert c.document_name == doc_name
        assert c.page_number >= 1
        assert c.index >= 0
        assert c.char_count == len(c.text)
        assert c.chunk_id
