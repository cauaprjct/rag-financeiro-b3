"""LLM-as-judge prompts for evaluation metrics."""

CONTEXT_PRECISION_PROMPT = """Avalie se o contexto recuperado é relevante para responder à pergunta.

Pergunta: {question}
Contexto: {context}

O contexto é relevante e útil para responder à pergunta?
Responda APENAS com um número de 0 a 1 (0 = irrelevante, 1 = totalmente relevante).
Resposta:"""


FAITHFULNESS_PROMPT = """Avalie se a resposta está fundamentada exclusivamente
no contexto fornecido, sem inventar informações.

Contexto: {context}
Resposta: {answer}

A resposta é fiel ao contexto (sem alucinações)?
Responda APENAS com um número de 0 a 1 (0 = não fiel, 1 = totalmente fiel).
Resposta:"""
