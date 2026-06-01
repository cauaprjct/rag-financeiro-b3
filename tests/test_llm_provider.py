"""Testes de exemplo do LLM_Provider."""

import pytest

from core.llm_provider import (
    LLMProvider,
    LLMTimeoutError,
    LLMRateLimitError,
    _safe_error,
)
from tests.fakes.fake_llm import FakeLLM


def test_fake_llm_satisfies_protocol():
    llm = FakeLLM()
    assert isinstance(llm, LLMProvider)


def test_timeout_raises_llm_timeout_error():
    """Req 6.5: timeout → indisponível."""
    llm = FakeLLM(raise_on_call=LLMTimeoutError("timeout"))
    with pytest.raises(LLMTimeoutError):
        llm.complete("pergunta")


def test_rate_limit_raises_llm_rate_limit_error():
    """Req 6.6: 429 → limite atingido."""
    llm = FakeLLM(raise_on_call=LLMRateLimitError("rate limit"))
    with pytest.raises(LLMRateLimitError):
        llm.complete("pergunta")


def test_api_key_never_in_error_message():
    """Req 9.6: chave nunca em mensagem de erro."""
    api_key = "sk-secret-key-12345"
    msg = f"Authentication failed with key {api_key} on server"
    safe = _safe_error(msg, api_key)
    assert api_key not in safe
    assert "***" in safe


def test_fake_llm_records_calls():
    llm = FakeLLM(response="ok")
    llm.complete("prompt1")
    llm.complete("prompt2")
    assert llm.calls == ["prompt1", "prompt2"]
