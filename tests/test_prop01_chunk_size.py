"""Property 1: Nenhum chunk excede o tamanho máximo."""

import tiktoken
from hypothesis import given, settings, assume
from hypothesis.strategies import text, integers

from core.chunking import chunk_document
from core.models import PageText

_ENC = tiktoken.get_encoding("cl100k_base")


@settings(max_examples=100)
@given(
    content=text(min_size=1, max_size=2000),
    chunk_size=integers(min_value=1, max_value=500),
    overlap=integers(min_value=0, max_value=499),
)
def test_no_chunk_exceeds_max_tokens(content, chunk_size, overlap):
    assume(overlap < chunk_size)
    pages = [PageText(page_number=1, text=content)]
    chunks = chunk_document(pages, "doc.pdf", chunk_size, overlap)
    for c in chunks:
        token_count = len(_ENC.encode(c.text))
        assert token_count <= chunk_size
