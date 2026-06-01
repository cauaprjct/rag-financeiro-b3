"""Testes de exemplo do Evaluation_Module e load_eval_dataset."""

from pathlib import Path

import pytest

from core.evaluation import EvaluationModule, load_eval_dataset, EvalDatasetError
from core.models import EvalExample
from tests.fakes.fake_llm import FakeLLM
from tests.fakes.hash_embeddings import HashEmbeddings

_FIXTURES = Path(__file__).parent / "fixtures"


def _module(tmp_path, llm=None):
    llm = llm or FakeLLM(response="0.8")
    return EvaluationModule(llm, HashEmbeddings(dimension=32), eval_dir=str(tmp_path))


class TestEvaluationModule:
    def test_empty_dataset_zero_metrics_and_message(self, tmp_path):
        """Req 7.4, 7.6: dataset vazio → métricas 0.0 + mensagem + persiste."""
        module = _module(tmp_path)
        report = module.evaluate([], answer_fn=lambda q: ("a", "c"))
        assert report.total == 0
        assert report.evaluated == 0
        assert report.context_precision == 0.0
        assert "vazio" in report.message.lower()
        # Persisted
        assert list(tmp_path.glob("report_*.json"))

    def test_failing_example_continues(self, tmp_path):
        """Req 7.9: exemplo que falha marcado não avaliado e segue."""
        module = _module(tmp_path)

        def answer_fn(q):
            if q == "q1":
                raise RuntimeError("pipeline failure")
            return ("answer", "context")

        dataset = [
            EvalExample(question="q0", reference_answer="r0"),
            EvalExample(question="q1", reference_answer="r1"),
            EvalExample(question="q2", reference_answer="r2"),
        ]
        report = module.evaluate(dataset, answer_fn)
        assert report.total == 3
        assert report.evaluated == 2
        assert any(not m.evaluated for m in report.examples)

    def test_no_example_evaluated_message(self, tmp_path):
        """Req 7.10: nenhum avaliado → mensagem dedicada."""
        module = _module(tmp_path)

        def answer_fn(q):
            raise RuntimeError("always fails")

        dataset = [EvalExample(question="q0", reference_answer="r0")]
        report = module.evaluate(dataset, answer_fn)
        assert report.evaluated == 0
        assert "Nenhum exemplo" in report.message

    def test_model_ids_registered(self, tmp_path):
        """Req 7.7: model ids registrados."""
        module = _module(tmp_path)
        report = module.evaluate(
            [EvalExample(question="q", reference_answer="r")],
            answer_fn=lambda q: ("a", "c"),
        )
        assert report.embedding_model_id == "fake/hash-embeddings"
        assert report.llm_model_id == "fake/llm"


class TestLoadEvalDataset:
    def test_valid_parse(self):
        """Req 7.11: parse válido."""
        examples = load_eval_dataset(_FIXTURES / "eval_valid.json")
        assert len(examples) == 2
        assert examples[0].question == "Qual foi a receita líquida no trimestre?"
        assert examples[0].reference_answer == "A receita líquida foi de R$ 1 bilhão."
        assert examples[0].expected_context == "Receita líquida trimestral"
        assert examples[1].expected_context == ""

    def test_missing_file_raises_with_path(self):
        """Req 7.12: arquivo ausente → EvalDatasetError com caminho."""
        with pytest.raises(EvalDatasetError, match="não encontrado"):
            load_eval_dataset(_FIXTURES / "ghost_dataset.json")

    def test_malformed_json_raises(self):
        """Req 7.13: JSON malformado → EvalDatasetError."""
        with pytest.raises(EvalDatasetError, match="malformado"):
            load_eval_dataset(_FIXTURES / "eval_malformed.json")

    def test_missing_required_field_raises(self):
        """Req 7.13: campo obrigatório ausente → EvalDatasetError."""
        with pytest.raises(EvalDatasetError, match="reference_answer"):
            load_eval_dataset(_FIXTURES / "eval_missing_field.json")
