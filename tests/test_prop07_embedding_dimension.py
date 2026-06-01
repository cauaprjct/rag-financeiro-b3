"""Property 7: Dimensão de embedding fixa e positiva."""

from hypothesis import given, settings
from hypothesis.strategies import text, integers, lists, characters

from tests.fakes.hash_embeddings import HashEmbeddings

_safe_text = text(
    alphabet=characters(blacklist_categories=("Cs",)),
    min_size=1,
    max_size=200,
)


@settings(max_examples=100)
@given(
    dimension=integers(min_value=1, max_value=512),
    passages=lists(_safe_text, min_size=1, max_size=10),
)
def test_embedding_dimension_fixed_positive(dimension, passages):
    emb = HashEmbeddings(dimension=dimension)
    assert emb.dimension == dimension
    assert emb.dimension > 0

    result = emb.embed_passages(passages)
    for vec in result.embeddings:
        assert vec is not None
        assert len(vec) == dimension

    # Query also has same dimension
    q_vec = emb.embed_query(passages[0])
    assert q_vec is not None
    assert len(q_vec) == dimension
