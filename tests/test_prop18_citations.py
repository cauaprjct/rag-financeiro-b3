"""Property 18: Citações correspondem aos chunks utilizados."""

from hypothesis import given, settings
from hypothesis.strategies import text, lists, integers, characters

from core.generation import build_citations
from core.models import Chunk, ScoredChunk

_safe_text = text(alphabet=characters(blacklist_categories=("Cs",)), min_size=3, max_size=300)


@settings(max_examples=100)
@given(
    chunk_texts=lists(_safe_text, min_size=1, max_size=8),
    pages=lists(integers(min_value=1, max_value=50), min_size=1, max_size=8),
)
def test_citations_match_chunks(chunk_texts, pages):
    # Align lengths
    n = min(len(chunk_texts), len(pages))
    scored = [
        ScoredChunk(
            chunk=Chunk(
                chunk_id=f"c{i}",
                text=chunk_texts[i],
                document_name=f"doc{i}.pdf",
                page_number=pages[i],
                index=i,
                char_count=len(chunk_texts[i]),
            ),
            score=0.8,
        )
        for i in range(n)
    ]

    citations = build_citations(scored)

    # One citation per chunk
    assert len(citations) == n

    for i, cit in enumerate(citations):
        assert cit.document_name == f"doc{i}.pdf"
        assert cit.page_number == pages[i]
        # Snippet is prefix of chunk text (max 200 chars)
        assert chunk_texts[i].startswith(cit.snippet) or cit.snippet == chunk_texts[i][:200]
