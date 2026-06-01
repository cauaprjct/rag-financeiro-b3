"""LLM Provider: Protocol, errors, GroqProvider, GeminiProvider, factory."""

from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)


class LLMError(Exception):
    pass


class LLMTimeoutError(LLMError):
    pass


class LLMRateLimitError(LLMError):
    pass


@runtime_checkable
class LLMProvider(Protocol):
    @property
    def model_id(self) -> str: ...

    def complete(self, prompt: str, *, timeout_s: float = 30.0) -> str: ...


class GroqProvider:
    """Groq API LLM provider."""

    def __init__(self, api_key: str, model_id: str = "llama-3.3-70b-versatile") -> None:
        self._api_key = api_key
        self._model_id = model_id

    @property
    def model_id(self) -> str:
        return self._model_id

    def complete(self, prompt: str, *, timeout_s: float = 30.0) -> str:
        from groq import Groq, APITimeoutError, RateLimitError

        try:
            client = Groq(api_key=self._api_key, timeout=timeout_s)
            response = client.chat.completions.create(
                model=self._model_id,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content or ""
        except APITimeoutError:
            raise LLMTimeoutError("Serviço indisponível: timeout na requisição")
        except RateLimitError:
            raise LLMRateLimitError("Limite de uso atingido: tente novamente mais tarde")
        except Exception as e:
            raise LLMError(f"Erro no LLM: {_safe_error(str(e), self._api_key)}")


class GeminiProvider:
    """Google Gemini API LLM provider."""

    def __init__(self, api_key: str, model_id: str = "gemini-2.5-flash") -> None:
        self._api_key = api_key
        self._model_id = model_id

    @property
    def model_id(self) -> str:
        return self._model_id

    def complete(self, prompt: str, *, timeout_s: float = 30.0) -> str:
        import google.generativeai as genai

        genai.configure(api_key=self._api_key)
        try:
            model = genai.GenerativeModel(self._model_id)
            response = model.generate_content(
                prompt,
                request_options={"timeout": timeout_s},
            )
            return response.text or ""
        except Exception as e:
            msg = str(e)
            if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                raise LLMRateLimitError("Limite de uso atingido: tente novamente mais tarde")
            if "timeout" in msg.lower() or "deadline" in msg.lower():
                raise LLMTimeoutError("Serviço indisponível: timeout na requisição")
            raise LLMError(f"Erro no LLM: {_safe_error(msg, self._api_key)}")


def make_llm_provider(cfg) -> LLMProvider:
    """Factory: create LLM provider from config."""
    if cfg.groq_api_key:
        return GroqProvider(api_key=cfg.groq_api_key)
    if cfg.gemini_api_key:
        return GeminiProvider(api_key=cfg.gemini_api_key)
    raise LLMError("Nenhuma chave de API configurada")


def _safe_error(msg: str, api_key: str) -> str:
    """Remove API key from error messages."""
    if api_key:
        return msg.replace(api_key, "***")
    return msg
