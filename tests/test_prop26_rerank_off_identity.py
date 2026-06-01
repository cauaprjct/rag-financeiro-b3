"""Property 26: Com reranking desabilitado, a saída é idêntica ao retrieval base."""

from hypothesis import given, settings
from hypothesis.strategies import lists, text, characters, integers

from core.models import Chunk
from core.retrieval import RetrievalModule, RetrievalParams
from tests.fakes.fake_cross_encoder import FakeCrossEncoder
from tests.fakes.hash_embeddings import HashEmbeddings
from tests.fakes.in_memory_store import InMemoryVectorStore

_safe_text = text(alphabet=characters(blacklist_categories=("Cs",)), min_size=5, max_size=100)


@settings(max_examples=100)
@given(
    texts=lists(_safe_text, min_size=3, max_size=10),
    top_k=integers(min_value=1, max_value=5),
)
def test_rerank_off_identical_to_base(texts, top_k):
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

    # Without reranker
    base_result = module.retrieve(
        RetrievalParams(query="consulta", top_k=top_k, rerank_enabled=False)
    )

    # With reranker object but disabled
    rerank_result = module.retrieve(
        RetrievalParams(
            query="consulta", top_k=top_k, reranker=FakeCrossEncoder(), rerank_enabled=False
        )
    )

    # Identical results
    base_ids = [(sc.chunk.chunk_id, sc.score) for sc in base_result.scored_chunks]
    rerank_ids = [(sc.chunk.chunk_id, sc.score) for sc in rerank_result.scored_chunks]
    assert base_ids == rerank_ids
