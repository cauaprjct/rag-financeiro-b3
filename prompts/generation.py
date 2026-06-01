"""Prompt template for grounded generation."""

GENERATION_PROMPT = """Responda à pergunta abaixo utilizando EXCLUSIVAMENTE as informações \
dos trechos fornecidos. Não utilize conhecimento externo.

## Pergunta
{question}

## Trechos
{chunks}

## Instruções
- Baseie sua resposta exclusivamente nos trechos acima.
- Se os trechos não contiverem informação suficiente, diga que não há informação disponível.
- Cite os trechos utilizados indicando o número entre colchetes [N].
"""


def format_chunks(chunks_texts: list[tuple[int, str]]) -> str:
    """Format numbered chunks for the prompt. Each tuple is (index, text)."""
    return "\n\n".join(f"[{idx}] {text}" for idx, text in chunks_texts)


def build_generation_prompt(question: str, chunks_texts: list[tuple[int, str]]) -> str:
    """Build the full generation prompt with question and chunks."""
    formatted = format_chunks(chunks_texts)
    return GENERATION_PROMPT.format(question=question, chunks=formatted)
