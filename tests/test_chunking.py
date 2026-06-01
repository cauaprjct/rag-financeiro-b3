"""Testes de exemplo do Chunking_Module."""

import math
import warnings

from core.chunking import chunk_document, normalize_params
from core.models import PageText


def test_overlap_gte_chunk_size_adjusted_with_warning():
    chunk_size = 100
    overlap = 150  # >= chunk_size

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        adjusted = normalize_params(chunk_size, overlap)

    assert adjusted == math.floor(0.10 * chunk_size)  # 10
    assert len(w) == 1
    assert "ajustado para" in str(w[0].message)


def test_overlap_equal_chunk_size_adjusted():
    chunk_size = 50
    overlap = 50

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        adjusted = normalize_params(chunk_size, overlap)

    assert adjusted == 5  # floor(0.10 * 50)
    assert len(w) == 1


def test_valid_overlap_unchanged():
    assert normalize_params(100, 20) == 20


def test_chunk_document_with_bad_overlap_still_works():
    pages = [PageText(page_number=1, text="Hello world this is a test document with some content.")]
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        chunks = chunk_document(pages, "doc.pdf", chunk_size=10, overlap=10)
    assert len(chunks) >= 1
