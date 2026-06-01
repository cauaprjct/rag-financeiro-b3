"""Property 8: Invariância ao tamanho do lote."""

from hypothesis import given, settings
from hypothesis.strategies import text, lists, characters

from tests.fakes.hash_embeddings import HashEmbeddings

_safe_text = text(
    alphabet=characters(blacklist_categories=("Cs",)),
    min_size=1,
    max_size=200,
)


@settings(max_examples=100)
@given(passages=lists(_safe_text, min_size=1, max_size=10))
def test_batch_invariance(passages):
    emb = HashEmbeddings(dimension=64)

    # Embed all at once
    batch_result = emb.embed_passages(passages)

    # Embed one by one
    individual_results = [emb.embed_passages([t]).embeddings[0] for t in passages]

    # Results must be identical regardless of batch size
    for batch_vec, ind_vec in zip(batch_result.embeddings, individual_results):
        assert batch_vec == ind_vec
