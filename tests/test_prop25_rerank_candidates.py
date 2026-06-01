"""Property 25: Reranking opera sobre no máximo N candidatos e retorna no máximo K."""

from hypothesis import given, settings, assume
from hypothesis.strategies import integers, lists, text, characters

from core.models import Chunk, ScoredChunk
from core.retrieval import RetrievalModule, RetrievalParams
from tests.fakes.hash_embeddings import HashEmbeddings
from tests.fakes.in_memory_store import InMemoryVectorStore

_safe_text = text(alphabet=characters(blacklist_categories=("Cs",)), min_size=5, max_size=100)


class TrackingReranker:
    """Reranker that tracks how many candidates it receives."""

    def __init__(self):
        self.received_count = 0

    def rerank(self, question: str, candidates: list[ScoredChunk], top_k: int) -> list[ScoredChunk]:
        self.received_count = len(candidates)
        # Simple: sort by existing score desc, truncate
        sorted_c = sorted(candidates, key=lambda s: s.score, reverse=True)
        return sorted_c[:top_k]


@settings(max_examples=100)
@given(
    texts=lists(_safe_text, min_size=5, max_size=20),
    top_k=integers(min_value=1, max_value=5),
    candidates_n=integers(min_value=5, max_value=20),
)
def test_rerank_candidates_n_and_k(texts, top_k, candidates_n):
    assume(candidates_n >= top_k)

    emb = HashEmbeddings(dimension=32)
    store = InMemoryVectorStore()
    reranker = TrackingReranker()

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
            query="consulta",
            top_k=top_k,
            reranker=reranker,
            rerank_enabled=True,
            rerank_candidates_n=candidates_n,
        )
    )

    # Reranker received at most N candidates
    assert reranker.received_count <= candidates_n

    # Result has at most K items
    assert len(result.scored_chunks) <= top_k
