"""Property 14: Resultados completos com pontuação normalizada."""

from hypothesis import given, settings
from hypothesis.strategies import lists, text, characters

from core.models import Chunk
from core.retrieval import RetrievalModule, RetrievalParams
from tests.fakes.hash_embeddings import HashEmbeddings
from tests.fakes.in_memory_store import InMemoryVectorStore

_safe_text = text(alphabet=characters(blacklist_categories=("Cs",)), min_size=5, max_size=100)


@settings(max_examples=100)
@given(texts=lists(_safe_text, min_size=2, max_size=10))
def test_results_complete_normalized(texts):
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
    result = module.retrieve(RetrievalParams(query="receita", top_k=5))

    for sc in result.scored_chunks:
        # Score normalized in [0, 1]
        assert 0.0 <= sc.score <= 1.0
        # Chunk is complete
        assert sc.chunk.chunk_id
        assert sc.chunk.text
        assert sc.chunk.document_name
        assert sc.chunk.page_number >= 1
