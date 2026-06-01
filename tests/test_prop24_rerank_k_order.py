"""Property 24: Reranking preserva o limite K e ordena por score de rerank decrescente."""

from hypothesis import given, settings
from hypothesis.strategies import integers, lists, text, characters

from core.models import Chunk
from core.retrieval import RetrievalModule, RetrievalParams
from tests.fakes.fake_cross_encoder import FakeCrossEncoder
from tests.fakes.hash_embeddings import HashEmbeddings
from tests.fakes.in_memory_store import InMemoryVectorStore

_safe_text = text(alphabet=characters(blacklist_categories=("Cs",)), min_size=5, max_size=100)


@settings(max_examples=100)
@given(
    texts=lists(_safe_text, min_size=5, max_size=15),
    top_k=integers(min_value=1, max_value=5),
)
def test_rerank_preserves_k_and_desc_order(texts, top_k):
    emb = HashEmbeddings(dimension=32)
    store = InMemoryVectorStore()
    reranker = FakeCrossEncoder()

    chunks = [
        Chunk(
            chunk_id=f"c{i}",
            text=t,
            document_name="doc.pdf",
            page_number=1,
            index=i,
            char_count=len(t),
        )
        for i, t in enumerate(texts)
    ]
    store.upsert_document("doc.pdf", chunks, emb.embed_passages(texts).embeddings)

    module = RetrievalModule(store, emb)
    result = module.retrieve(
        RetrievalParams(
            query="consulta financeira",
            top_k=top_k,
            reranker=reranker,
            rerank_enabled=True,
            rerank_candidates_n=len(texts),
        )
    )

    # Respects limit K
    assert len(result.scored_chunks) <= top_k

    # Descending order
    scores = [sc.score for sc in result.scored_chunks]
    assert scores == sorted(scores, reverse=True)
