"""Testes de exemplo do Generation_Module."""

from core.generation import GenerationModule
from core.llm_provider import LLMTimeoutError, LLMRateLimitError, LLMError
from core.models import Chunk, ScoredChunk
from prompts.generation import GENERATION_PROMPT
from tests.fakes.fake_llm import FakeLLM


def _make_chunks(n=2):
    return [
        ScoredChunk(
            chunk=Chunk(
                chunk_id=f"c{i}",
                text=f"Texto do chunk {i}",
                document_name="doc.pdf",
                page_number=i + 1,
                index=i,
                char_count=15,
            ),
            score=0.9 - i * 0.1,
        )
        for i in range(n)
    ]


def test_grounding_instruction_in_prompt():
    """Req 6.4: instrução de fundamentação exclusiva nos chunks."""
    assert "EXCLUSIVAMENTE" in GENERATION_PROMPT


def test_timeout_returns_unavailable():
    """Req 6.5: timeout → indisponível."""
    llm = FakeLLM(raise_on_call=LLMTimeoutError("timeout"))
    module = GenerationModule(llm)
    result = module.generate("pergunta", _make_chunks())
    assert result.status == "unavailable"


def test_rate_limit_returns_rate_limited():
    """Req 6.6: 429 → rate_limited."""
    llm = FakeLLM(raise_on_call=LLMRateLimitError("429"))
    module = GenerationModule(llm)
    result = module.generate("pergunta", _make_chunks())
    assert result.status == "rate_limited"


def test_llm_invoked_with_chunks():
    """Req 6.8: LLM invocado com chunks no prompt."""
    llm = FakeLLM(response="A receita foi de R$1bi.")
    module = GenerationModule(llm)
    chunks = _make_chunks()
    result = module.generate("Qual a receita?", chunks)

    assert result.status == "ok"
    assert result.answer == "A receita foi de R$1bi."
    assert len(result.citations) == 2
    # Prompt sent to LLM contains the chunk texts
    assert "Texto do chunk 0" in llm.calls[0]
    assert "Texto do chunk 1" in llm.calls[0]


def test_generic_error_no_internal_details():
    """Req 6.8: erro genérico não vaza detalhes internos."""
    llm = FakeLLM(raise_on_call=LLMError("Internal server error at 10.0.0.1"))
    module = GenerationModule(llm)
    result = module.generate("pergunta", _make_chunks())
    assert result.status == "unavailable"
    assert "10.0.0.1" not in result.answer
