"""Property 13: Retrieval respeita limite K e ordenação decrescente."""

from hypothesis import given, settings
from hypothesis.strategies import integers, lists, text, characters

from core.models import Chunk
from core.retrieval import RetrievalModule, RetrievalParams
from tests.fakes.hash_embeddings import HashEmbeddings
from tests.fakes.in_memory_store import InMemoryVectorStore

_safe_text = text(alphabet=characters(blacklist_categories=("Cs",)), min_size=5, max_size=100)


@settings(max_examples=100)
@given(
    texts=lists(_safe_text, min_size=3, max_size=15),
    top_k=integers(min_value=1, max_value=10),
)
def test_retrieval_respects_k_and_desc_order(texts, top_k):
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
    vectors = emb.embed_passages(texts).embeddings
    store.upsert_document("doc.pdf", chunks, vectors)

    module = RetrievalModule(store, emb)
    result = module.retrieve(RetrievalParams(query="consulta financeira", top_k=top_k))

    # Respects limit K
    assert len(result.scored_chunks) <= top_k

    # Descending order
    scores = [sc.score for sc in result.scored_chunks]
    assert scores == sorted(scores, reverse=True)
