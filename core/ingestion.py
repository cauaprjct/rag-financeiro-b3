"""Ingestion module: PDF validation, text extraction, table extraction."""

from __future__ import annotations

from pathlib import Path

import pdfplumber
import pypdf

from core.models import IngestionResult, PageText

_MAX_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB
_PDF_MAGIC = b"%PDF"
_MIN_CHARS_THRESHOLD = 10


def ingest_file(path: Path) -> IngestionResult:
    """Ingest a single PDF file. Returns IngestionResult with pages or error."""
    name = path.name

    # Validate existence
    if not path.exists():
        return IngestionResult(document_name=name, success=False, error="Arquivo não encontrado")

    # Validate magic bytes
    try:
        with open(path, "rb") as f:
            header = f.read(4)
    except OSError as e:
        return IngestionResult(document_name=name, success=False, error=str(e))

    if header[:4] != _PDF_MAGIC:
        return IngestionResult(document_name=name, success=False, error="Arquivo não é PDF válido")

    # Validate size
    size = path.stat().st_size
    if size > _MAX_SIZE_BYTES:
        return IngestionResult(
            document_name=name, success=False, error=f"Arquivo excede 50MB ({size} bytes)"
        )

    # Extract text
    try:
        pages = _extract_pages(path)
    except Exception as e:
        return IngestionResult(document_name=name, success=False, error=f"Erro ao extrair: {e}")

    # Check if meaningful text was extracted
    total_chars = sum(len(p.text) for p in pages)
    if total_chars < _MIN_CHARS_THRESHOLD:
        return IngestionResult(
            document_name=name,
            pages=pages,
            success=True,
            error=f"Pouco texto extraído ({total_chars} chars) - possível PDF só-imagem",
        )

    return IngestionResult(document_name=name, pages=pages, success=True)


def _extract_pages(path: Path) -> list[PageText]:
    """Extract text from each page, including tables via pdfplumber."""
    pages: list[PageText] = []

    # Text extraction with pypdf
    reader = pypdf.PdfReader(str(path))
    pypdf_texts = [page.extract_text() or "" for page in reader.pages]

    # Table extraction with pdfplumber
    with pdfplumber.open(str(path)) as pdf:
        for i, page in enumerate(pdf.pages):
            text_parts: list[str] = []

            # Base text from pypdf
            if i < len(pypdf_texts) and pypdf_texts[i].strip():
                text_parts.append(pypdf_texts[i])

            # Tables
            tables = page.extract_tables()
            for table in tables:
                if table:
                    rows = []
                    for row in table:
                        cells = [str(c) if c else "" for c in row]
                        rows.append(" | ".join(cells))
                    text_parts.append("\n".join(rows))

            combined = "\n".join(text_parts)
            pages.append(PageText(page_number=i + 1, text=combined))

    return pages


def ingest_batch(paths: list[Path]) -> list[IngestionResult]:
    """Ingest multiple files. Returns one IngestionResult per file, same order."""
    return [ingest_file(p) for p in paths]
