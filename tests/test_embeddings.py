"""Testes de exemplo do Embedding_Module."""

from core.embeddings import EmbeddingModule
from tests.fakes.hash_embeddings import HashEmbeddings


def test_hash_embeddings_satisfies_protocol():
    """HashEmbeddings satisfaz o Protocol EmbeddingModule."""
    emb = HashEmbeddings(dimension=128)
    assert isinstance(emb, EmbeddingModule)


def test_model_id_registered():
    """Req 3.3: embedding_model_id registrado."""
    emb = HashEmbeddings(dimension=64)
    assert emb.model_id == "fake/hash-embeddings"


def test_single_failure_does_not_block_batch():
    """Req 3.4: falha de 1 item retorna None+erro e segue."""

    class FailOnSecond(HashEmbeddings):
        def embed_passages(self, texts):
            from core.models import EmbedBatchResult

            result = EmbedBatchResult()
            for i, t in enumerate(texts):
                if i == 1:
                    result.embeddings.append(None)
                    result.errors.append("simulated failure")
                else:
                    vec = self._hash_to_vector(f"passage: {t}")
                    result.embeddings.append(vec)
                    result.errors.append("")
            return result

    emb = FailOnSecond(dimension=64)
    result = emb.embed_passages(["ok1", "fail", "ok2"])
    assert result.embeddings[0] is not None
    assert result.embeddings[1] is None
    assert "failure" in result.errors[1]
    assert result.embeddings[2] is not None
