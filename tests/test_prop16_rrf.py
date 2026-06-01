"""Property 16: Fusão híbrida (RRF) sem duplicatas, ordenada e limitada."""

from hypothesis import given, settings
from hypothesis.strategies import lists, integers, text, characters

from core.models import Chunk, ScoredChunk
from core.retrieval import reciprocal_rank_fusion

_safe_text = text(alphabet=characters(blacklist_categories=("Cs",)), min_size=3, max_size=50)


def _make_scored(texts: list[str], prefix: str) -> list[ScoredChunk]:
    return [
        ScoredChunk(
            chunk=Chunk(
                chunk_id=f"{prefix}{i}",
                text=t,
                document_name="d.pdf",
                page_number=1,
                index=i,
                char_count=len(t),
            ),
            score=1.0 / (i + 1),
        )
        for i, t in enumerate(texts)
    ]


@settings(max_examples=100)
@given(
    texts=lists(_safe_text, min_size=2, max_size=10),
    top_k=integers(min_value=1, max_value=10),
)
def test_rrf_no_duplicates_ordered_limited(texts, top_k):
    # Create two lists with overlapping chunks (same chunk_ids)
    list1 = _make_scored(texts, "c")
    list2 = _make_scored(texts, "c")  # Same IDs = overlap

    result = reciprocal_rank_fusion([list1, list2], top_k=top_k)

    # No duplicates
    ids = [sc.chunk.chunk_id for sc in result]
    assert len(ids) == len(set(ids))

    # Limited to top_k
    assert len(result) <= top_k

    # Ordered descending
    scores = [sc.score for sc in result]
    assert scores == sorted(scores, reverse=True)

    # Normalized to [0, 1]
    for sc in result:
        assert 0.0 <= sc.score <= 1.0


@settings(max_examples=100)
@given(
    texts_a=lists(_safe_text, min_size=1, max_size=5),
    texts_b=lists(_safe_text, min_size=1, max_size=5),
    top_k=integers(min_value=1, max_value=10),
)
def test_rrf_disjoint_lists_no_duplicates(texts_a, texts_b, top_k):
    list1 = _make_scored(texts_a, "a")
    list2 = _make_scored(texts_b, "b")

    result = reciprocal_rank_fusion([list1, list2], top_k=top_k)

    ids = [sc.chunk.chunk_id for sc in result]
    assert len(ids) == len(set(ids))
    assert len(result) <= top_k
