"""Generate test PDF fixtures for ingestion tests."""

from pathlib import Path
from fpdf import FPDF

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def generate_valid_pdf() -> None:
    """Single page PDF with text."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    pdf.cell(text="Relatorio Financeiro B3 2024 - Dados trimestrais consolidados")
    pdf.output(str(FIXTURES_DIR / "valid.pdf"))


def generate_multipage_pdf() -> None:
    """Two page PDF."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    pdf.cell(text="Pagina 1 - Receita operacional liquida")
    pdf.add_page()
    pdf.cell(text="Pagina 2 - Despesas administrativas")
    pdf.output(str(FIXTURES_DIR / "multipage.pdf"))


def generate_corrupted_pdf() -> None:
    """File with PDF magic but corrupted content."""
    with open(FIXTURES_DIR / "corrupted.pdf", "wb") as f:
        f.write(b"%PDF-1.4\n%%garbage corrupted content here\n")


def generate_image_only_pdf() -> None:
    """PDF with no extractable text."""
    pdf = FPDF()
    pdf.add_page()
    # No text added - blank page
    pdf.output(str(FIXTURES_DIR / "image_only.pdf"))


def generate_not_pdf() -> None:
    """A text file pretending to be PDF."""
    with open(FIXTURES_DIR / "not_a_pdf.txt", "w") as f:
        f.write("This is not a PDF file at all.")


def generate_all() -> None:
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    generate_valid_pdf()
    generate_multipage_pdf()
    generate_corrupted_pdf()
    generate_image_only_pdf()
    generate_not_pdf()
    print(f"Fixtures generated in {FIXTURES_DIR}")


if __name__ == "__main__":
    generate_all()
