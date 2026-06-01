"""Generation module: build prompt, citations, generate answer."""

from __future__ import annotations

from dataclasses import dataclass, field

from core.llm_provider import LLMError, LLMTimeoutError, LLMRateLimitError
from core.models import Citation, ScoredChunk
from prompts.generation import build_generation_prompt


@dataclass
class GenerationResult:
    answer: str = ""
    citations: list[Citation] = field(default_factory=list)
    status: str = "ok"  # ok | no_context | unavailable | rate_limited


def build_citations(scored_chunks: list[ScoredChunk]) -> list[Citation]:
    """Build one Citation per chunk: document, page, snippet."""
    citations = []
    for sc in scored_chunks:
        snippet = sc.chunk.text[:200]
        citations.append(
            Citation(
                document_name=sc.chunk.document_name,
                page_number=sc.chunk.page_number,
                snippet=snippet,
            )
        )
    return citations


def build_prompt(question: str, scored_chunks: list[ScoredChunk]) -> str:
    """Build the generation prompt from question and scored chunks."""
    chunks_texts = [(i + 1, sc.chunk.text) for i, sc in enumerate(scored_chunks)]
    return build_generation_prompt(question, chunks_texts)


class GenerationModule:
    """Generates grounded answers with citations."""

    def __init__(self, llm_provider) -> None:
        self._llm = llm_provider

    def generate(self, question: str, scored_chunks: list[ScoredChunk]) -> GenerationResult:
        # No context → don't invoke LLM
        if not scored_chunks:
            return GenerationResult(
                answer="Informação insuficiente para responder à pergunta.",
                status="no_context",
            )

        prompt = build_prompt(question, scored_chunks)
        citations = build_citations(scored_chunks)

        try:
            answer = self._llm.complete(prompt)
        except LLMTimeoutError:
            return GenerationResult(
                answer="Serviço temporariamente indisponível.", status="unavailable"
            )
        except LLMRateLimitError:
            return GenerationResult(
                answer="Limite de uso atingido. Tente novamente mais tarde.", status="rate_limited"
            )
        except LLMError:
            return GenerationResult(
                answer="Serviço temporariamente indisponível.", status="unavailable"
            )

        return GenerationResult(answer=answer, citations=citations, status="ok")
