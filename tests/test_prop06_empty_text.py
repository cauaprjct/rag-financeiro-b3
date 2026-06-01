"""Property 6: Texto vazio ou em branco não gera chunks."""

from hypothesis import given, settings
from hypothesis.strategies import integers, sampled_from

from core.chunking import chunk_document
from core.models import PageText


@settings(max_examples=100)
@given(
    content=sampled_from(["", " ", "  ", "\n", "\t", "   \n  "]),
    chunk_size=integers(min_value=1, max_value=500),
)
def test_empty_text_no_chunks(content, chunk_size):
    pages = [PageText(page_number=1, text=content)]
    chunks = chunk_document(pages, "doc.pdf", chunk_size, overlap=0)
    assert chunks == []
