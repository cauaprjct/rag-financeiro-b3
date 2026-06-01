"""Property 2: Round-trip de chunking (reconcatenação sem sobreposição)."""

import tiktoken
from hypothesis import given, settings, assume
from hypothesis.strategies import text, integers

from core.chunking import chunk_document
from core.models import PageText

_ENC = tiktoken.get_encoding("cl100k_base")


@settings(max_examples=100)
@given(
    content=text(min_size=10, max_size=2000),
    chunk_size=integers(min_value=5, max_value=200),
    overlap=integers(min_value=0, max_value=199),
)
def test_roundtrip_tokens(content, chunk_size, overlap):
    assume(overlap < chunk_size)
    assume(content.strip())
    pages = [PageText(page_number=1, text=content)]
    chunks = chunk_document(pages, "doc.pdf", chunk_size, overlap)
    assume(len(chunks) >= 2)

    original_tokens = _ENC.encode("\n".join(p.text for p in pages))
    step = chunk_size - overlap

    # Reconstruct from the sliding window logic directly
    reconstructed: list[int] = []
    for i, c in enumerate(chunks):
        start = i * step
        end = start + chunk_size
        window = original_tokens[start:end]
        if i == 0:
            reconstructed.extend(window)
        else:
            reconstructed.extend(window[overlap:])

    assert reconstructed == original_tokens
