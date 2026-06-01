"""Property 5: Texto curto gera um único chunk."""

import tiktoken
from hypothesis import given, settings, assume
from hypothesis.strategies import text, integers

from core.chunking import chunk_document
from core.models import PageText

_ENC = tiktoken.get_encoding("cl100k_base")


@settings(max_examples=100)
@given(
    content=text(min_size=1, max_size=500),
    chunk_size=integers(min_value=10, max_value=1000),
)
def test_short_text_single_chunk(content, chunk_size):
    assume(content.strip())
    tokens = _ENC.encode(content)
    assume(0 < len(tokens) <= chunk_size)
    pages = [PageText(page_number=1, text=content)]
    chunks = chunk_document(pages, "doc.pdf", chunk_size, overlap=0)
    assert len(chunks) == 1
