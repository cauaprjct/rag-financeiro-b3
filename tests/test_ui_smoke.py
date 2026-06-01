"""Smoke tests da UI com streamlit.testing.v1.AppTest."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

from core.generation import GenerationResult
from core.models import Chunk, Citation
from tests.fakes.hash_embeddings import HashEmbeddings
from tests.fakes.in_memory_store import InMemoryVectorStore

_APP = str(Path(__file__).parent.parent / "app.py")


class FakePipeline:
    """Fake pipeline injected into session_state for UI tests."""

    def __init__(self, *, status="ok"):
        self.store = InMemoryVectorStore()
        self.last_document_filter = "UNSET"
        self._status = status
        self._seed()

    def _seed(self):
        emb = HashEmbeddings(dimension=32)
        for name in ("a.pdf", "b.pdf"):
            chunks = [
                Chunk(
                    chunk_id=f"{name}0",
                    text=f"texto {name}",
                    document_name=name,
                    page_number=1,
                    index=0,
                    char_count=10,
                )
            ]
            self.store.upsert_document(
                name, chunks, emb.embed_passages([f"texto {name}"]).embeddings
            )

    def query(self, question, *, document_filter=None):
        self.last_document_filter = document_filter
        if self._status != "ok":
            return GenerationResult(
                answer="Serviço temporariamente indisponível.", status=self._status
            )
        return GenerationResult(
            answer="Resposta fake.",
            citations=[Citation(document_name="a.pdf", page_number=1, snippet="trecho")],
            status="ok",
        )


def _app_with_pipeline(pipeline):
    at = AppTest.from_file(_APP)
    at.session_state["pipeline"] = pipeline
    return at.run()


def test_disclaimer_present():
    """Req 10.1: disclaimer permanente."""
    at = _app_with_pipeline(FakePipeline())
    captions = " ".join(c.value for c in at.caption)
    assert "recomendação de investimento" in captions


def test_empty_question_warning():
    """Req 8.6: pergunta vazia gera aviso e não consulta."""
    at = _app_with_pipeline(FakePipeline())
    # Click "Perguntar" with empty question
    ask_btn = [b for b in at.button if b.label == "Perguntar"][0]
    ask_btn.click().run()
    assert len(at.warning) >= 1
    assert at.session_state.pipeline.last_document_filter == "UNSET"  # query not called


def test_generation_error_shows_message_and_preserves_session():
    """Req 8.12: erro de geração exibe mensagem preservando sessão."""
    at = _app_with_pipeline(FakePipeline(status="unavailable"))
    at.text_input[0].set_value("Qual a receita?").run()
    ask_btn = [b for b in at.button if b.label == "Perguntar"][0]
    ask_btn.click().run()
    assert len(at.error) >= 1
    # History still recorded (session preserved)
    assert len(at.session_state.history) == 1


def test_document_list_updates_on_remove():
    """Req 8.11: lista atualiza ao remover documento."""
    pipeline = FakePipeline()
    at = _app_with_pipeline(pipeline)
    assert pipeline.store.count() == 2
    # Click delete button for a.pdf
    del_btn = [b for b in at.button if b.key == "del_a.pdf"][0]
    del_btn.click().run()
    assert pipeline.store.delete_document("a.pdf").deleted_count == 0  # already deleted
    assert "a.pdf" not in [d.name for d in pipeline.store.list_documents()]


def test_multiselect_forwards_document_filter():
    """Req 8.13: seleção de subconjunto encaminha document_filter."""
    pipeline = FakePipeline()
    at = _app_with_pipeline(pipeline)
    at.multiselect[0].select("a.pdf").run()
    at.text_input[0].set_value("pergunta").run()
    ask_btn = [b for b in at.button if b.label == "Perguntar"][0]
    ask_btn.click().run()
    assert pipeline.last_document_filter == ["a.pdf"]


def test_empty_selection_considers_all():
    """Req 8.14: seleção vazia considera todos (document_filter=None)."""
    pipeline = FakePipeline()
    at = _app_with_pipeline(pipeline)
    at.text_input[0].set_value("pergunta").run()
    ask_btn = [b for b in at.button if b.label == "Perguntar"][0]
    ask_btn.click().run()
    assert pipeline.last_document_filter is None
