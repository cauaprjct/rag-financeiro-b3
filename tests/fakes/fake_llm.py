"""Fake LLM for offline testing. Records calls, can simulate errors."""

from __future__ import annotations


class FakeLLM:
    """Fake LLM that records calls and returns canned responses."""

    def __init__(
        self, response: str = "Resposta fake do LLM.", *, raise_on_call: Exception | None = None
    ) -> None:
        self._response = response
        self._raise_on_call = raise_on_call
        self.calls: list[str] = []

    @property
    def model_id(self) -> str:
        return "fake/llm"

    def complete(self, prompt: str, *, timeout_s: float = 30.0) -> str:
        self.calls.append(prompt)
        if self._raise_on_call:
            raise self._raise_on_call
        return self._response
