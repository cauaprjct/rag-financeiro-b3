"""Testes de exemplo de ingestão (casos de borda)."""

from pathlib import Path

from core.ingestion import ingest_file, ingest_batch

_FIXTURES = Path(__file__).parent / "fixtures"


def test_not_pdf_rejected():
    """Req 1.3: não-PDF rejeitado."""
    result = ingest_file(_FIXTURES / "not_a_pdf.txt")
    assert result.success is False
    assert "não é PDF" in result.error


def test_corrupted_pdf_does_not_block_batch():
    """Req 1.4: corrompido não impede demais."""
    paths = [_FIXTURES / "corrupted.pdf", _FIXTURES / "valid.pdf"]
    results = ingest_batch(paths)
    assert len(results) == 2
    # Corrupted may fail but valid should succeed
    assert results[1].success is True


def test_image_only_pdf_warning():
    """Req 1.5: PDF só-imagem com <10 chars gera aviso."""
    result = ingest_file(_FIXTURES / "image_only.pdf")
    assert result.success is True
    assert "só-imagem" in result.error or "Pouco texto" in result.error


def test_valid_pdf_extracts_text():
    """Req 1.2: PDF válido extrai texto."""
    result = ingest_file(_FIXTURES / "valid.pdf")
    assert result.success is True
    assert len(result.pages) >= 1
    total_text = "".join(p.text for p in result.pages)
    assert len(total_text) >= 10


def test_multipage_pdf():
    """Req 1.2: multipágina extrai todas as páginas."""
    result = ingest_file(_FIXTURES / "multipage.pdf")
    assert result.success is True
    assert len(result.pages) == 2


def test_exceeds_50mb(tmp_path):
    """Req 1.9: arquivo >50MB rejeitado."""
    big_file = tmp_path / "big.pdf"
    # Write PDF magic + enough data to exceed 50MB
    with open(big_file, "wb") as f:
        f.write(b"%PDF-1.4\n")
        f.write(b"x" * (50 * 1024 * 1024 + 1))
    result = ingest_file(big_file)
    assert result.success is False
    assert "50MB" in result.error


def test_nonexistent_file():
    """Arquivo inexistente retorna erro."""
    result = ingest_file(Path("/tmp/nonexistent_xyz.pdf"))
    assert result.success is False
    assert "não encontrado" in result.error
