"""Property 17: Prompt contém a pergunta e os trechos recuperados."""

from hypothesis import given, settings
from hypothesis.strategies import text, lists, characters

from core.generation import build_prompt
from core.models import Chunk, ScoredChunk

_safe_text = text(alphabet=characters(blacklist_categories=("Cs",)), min_size=3, max_size=200)


@settings(max_examples=100)
@given(
    question=_safe_text,
    chunk_texts=lists(_safe_text, min_size=1, max_size=5),
)
def test_prompt_contains_question_and_chunks(question, chunk_texts):
    scored = [
        ScoredChunk(
            chunk=Chunk(
                chunk_id=f"c{i}",
                text=t,
                document_name="d.pdf",
                page_number=1,
                index=i,
                char_count=len(t),
            ),
            score=0.9,
        )
        for i, t in enumerate(chunk_texts)
    ]

    prompt = build_prompt(question, scored)

    # Question is in the prompt
    assert question in prompt

    # Each chunk text is in the prompt
    for t in chunk_texts:
        assert t in prompt
