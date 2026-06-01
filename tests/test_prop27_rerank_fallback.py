"""Property 27: Falha do reranker resulta em fallback para o retrieval base sem propagar exceção."""

from hypothesis import given, settings
from hypothesis.strategies import lists, text, characters, integers

from core.models import Chunk, ScoredChunk
from core.retrieval import RetrievalModule, RetrievalParams
from tests.fakes.hash_embeddings import HashEmbeddings
from tests.fakes.in_memory_store import InMemoryVectorStore

_safe_text = text(alphabet=characters(blacklist_categories=("Cs",)), min_size=5, max_size=100)


class ExplodingReranker:
    """Reranker that always raises an exception."""

    def rerank(self, question: str, candidates: list[ScoredChunk], top_k: int) -> list[ScoredChunk]:
        raise RuntimeError("Reranker exploded!")


@settings(max_examples=100)
@given(
    texts=lists(_safe_text, min_size=3, max_size=10),
    top_k=integers(min_value=1, max_value=5),
)
def test_reranker_failure_fallback_no_exception(texts, top_k):
    emb = HashEmbeddings(dimension=32)
    store = InMemoryVectorStore()

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

    # Should NOT raise, should fallback gracefully
    result = module.retrieve(
        RetrievalParams(
            query="consulta",
            top_k=top_k,
            reranker=ExplodingReranker(),
            rerank_enabled=True,
            rerank_candidates_n=len(texts),
        )
    )

    # Returns results (fallback to base top-K)
    assert len(result.scored_chunks) <= top_k
    # No exception propagated - test passes if we reach here
