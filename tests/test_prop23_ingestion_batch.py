"""Property 23: Ingestão em lote produz um resultado por arquivo."""

from pathlib import Path
from hypothesis import given, settings
from hypothesis.strategies import lists, sampled_from

from core.ingestion import ingest_batch

_FIXTURES = Path(__file__).parent / "fixtures"
_FILES = [
    _FIXTURES / "valid.pdf",
    _FIXTURES / "multipage.pdf",
    _FIXTURES / "corrupted.pdf",
    _FIXTURES / "image_only.pdf",
    _FIXTURES / "not_a_pdf.txt",
    _FIXTURES / "nonexistent.pdf",
]


@settings(max_examples=100)
@given(paths=lists(sampled_from(_FILES), min_size=0, max_size=20))
def test_batch_one_result_per_file_same_order(paths):
    results = ingest_batch(paths)
    # Exactly one result per file
    assert len(results) == len(paths)
    # Same order
    for r, p in zip(results, paths):
        assert r.document_name == p.name
