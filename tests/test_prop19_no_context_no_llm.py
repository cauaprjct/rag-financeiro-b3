"""Property 19: Ausência de contexto não invoca o LLM."""

from hypothesis import given, settings
from hypothesis.strategies import text, characters

from core.generation import GenerationModule
from tests.fakes.fake_llm import FakeLLM

_safe_text = text(alphabet=characters(blacklist_categories=("Cs",)), min_size=1, max_size=200)


@settings(max_examples=100)
@given(question=_safe_text)
def test_no_context_no_llm_call(question):
    llm = FakeLLM()
    module = GenerationModule(llm)

    result = module.generate(question, scored_chunks=[])

    # LLM never called
    assert llm.calls == []
    # Status indicates no context
    assert result.status == "no_context"
    # No citations
    assert result.citations == []
