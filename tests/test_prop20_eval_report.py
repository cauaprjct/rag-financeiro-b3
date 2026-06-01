"""Property 20: Invariantes do relatório de avaliação."""

from hypothesis import given, settings
from hypothesis.strategies import text, characters, floats, integers

from core.evaluation import EvaluationModule
from core.models import EvalExample
from tests.fakes.fake_llm import FakeLLM
from tests.fakes.hash_embeddings import HashEmbeddings

_safe_text = text(alphabet=characters(blacklist_categories=("Cs",)), min_size=1, max_size=50)


@settings(max_examples=100, deadline=None)
@given(
    n_examples=integers(min_value=0, max_value=10),
    judge_score=floats(min_value=0.0, max_value=1.0),
)
def test_report_invariants(tmp_path_factory, n_examples, judge_score):
    tmp = tmp_path_factory.mktemp("eval")
    llm = FakeLLM(response=str(judge_score))
    emb = HashEmbeddings(dimension=32)
    module = EvaluationModule(llm, emb, eval_dir=str(tmp))

    dataset = [EvalExample(question=f"q{i}", reference_answer=f"ref{i}") for i in range(n_examples)]

    report = module.evaluate(dataset, answer_fn=lambda q: (f"answer for {q}", f"context for {q}"))

    # 0 <= evaluated <= total
    assert 0 <= report.evaluated <= report.total
    assert report.total == n_examples

    # Metrics in [0, 1]
    assert 0.0 <= report.context_precision <= 1.0
    assert 0.0 <= report.faithfulness <= 1.0
    assert 0.0 <= report.response_relevancy <= 1.0

    # Per-example metrics in [0, 1]
    for m in report.examples:
        assert 0.0 <= m.context_precision <= 1.0
        assert 0.0 <= m.faithfulness <= 1.0
        assert 0.0 <= m.response_relevancy <= 1.0
