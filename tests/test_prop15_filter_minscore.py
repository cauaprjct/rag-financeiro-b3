"""Property 15: Resultados respeitam filtro de documentos e pontuação mínima."""

from hypothesis import given, settings
from hypothesis.strategies import lists, text, characters, floats

from core.models import Chunk
from core.retrieval import RetrievalModule, RetrievalParams
from tests.fakes.hash_embeddings import HashEmbeddings
from tests.fakes.in_memory_store import InMemoryVectorStore

_safe_text = text(alphabet=characters(blacklist_categories=("Cs",)), min_size=5, max_size=100)


@settings(max_examples=100)
@given(
    texts_a=lists(_safe_text, min_size=2, max_size=5),
    texts_b=lists(_safe_text, min_size=2, max_size=5),
)
def test_document_filter_respected(texts_a, texts_b):
    emb = HashEmbeddings(dimension=32)
    store = InMemoryVectorStore()

    chunks_a = [
        Chunk(
            chunk_id=f"a{i}",
            text=t,
            document_name="a.pdf",
            page_number=1,
            index=i,
            char_count=len(t),
        )
        for i, t in enumerate(texts_a)
    ]
    chunks_b = [
        Chunk(
            chunk_id=f"b{i}",
            text=t,
            document_name="b.pdf",
            page_number=1,
            index=i,
            char_count=len(t),
        )
        for i, t in enumerate(texts_b)
    ]
    store.upsert_document("a.pdf", chunks_a, emb.embed_passages(texts_a).embeddings)
    store.upsert_document("b.pdf", chunks_b, emb.embed_passages(texts_b).embeddings)

    module = RetrievalModule(store, emb)
    result = module.retrieve(RetrievalParams(query="consulta", top_k=10, document_filter=["a.pdf"]))

    for sc in result.scored_chunks:
        assert sc.chunk.document_name == "a.pdf"


@settings(max_examples=100)
@given(min_score=floats(min_value=0.01, max_value=0.99))
def test_min_relevance_score_filter(min_score):
    emb = HashEmbeddings(dimension=32)
    store = InMemoryVectorStore()

    texts = ["receita líquida trimestral", "despesa operacional", "lucro bruto", "ativo total"]
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
        RetrievalParams(query="receita", top_k=10, min_relevance_score=min_score)
    )

    for sc in result.scored_chunks:
        assert sc.score >= min_score
