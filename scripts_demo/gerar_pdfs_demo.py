"""Gera PDFs de demonstracao realistas (estilo release de resultados B3) para o screenshot."""
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle
from reportlab.lib import colors

OUT = Path(__file__).parent.parent / "data" / "sample_docs"
OUT.mkdir(parents=True, exist_ok=True)


def build_petr4():
    doc = SimpleDocTemplate(
        str(OUT / "PETR4_Release_Resultados_4T2025.pdf"),
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
        title="PETR4 - Release de Resultados 4T2025",
    )
    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("H1", parent=styles["Heading1"], textColor=colors.HexColor("#003366"))
    body = styles["BodyText"]
    body.spaceAfter = 8
    flow = []

    flow.append(Paragraph("PETROBRAS (PETR4) - Release de Resultados 4T2025", h1))
    flow.append(Paragraph("Documento de divulgacao trimestral - B3 / CVM", body))
    flow.append(Spacer(1, 0.4 * cm))

    flow.append(Paragraph("<b>Destaques Financeiros do Trimestre</b>", styles["Heading2"]))
    flow.append(Paragraph(
        "A Petrobras encerrou o quarto trimestre de 2025 com receita liquida consolidada de "
        "R$ 142,7 bilhoes, alta de 8,3% em relacao ao mesmo periodo do ano anterior. "
        "O EBITDA ajustado atingiu R$ 68,4 bilhoes, com margem de 47,9%. "
        "O lucro liquido atribuivel aos acionistas foi de R$ 31,2 bilhoes no trimestre.",
        body,
    ))

    flow.append(Paragraph("<b>Producao e Vendas</b>", styles["Heading2"]))
    flow.append(Paragraph(
        "A producao total de oleo e gas natural alcancou 2,82 milhoes de barris de oleo "
        "equivalente por dia (boed) no 4T2025, crescimento de 4,1% sobre o 4T2024. "
        "A producao do pre-sal representou 79% do total, com destaque para os campos de "
        "Buzios, Tupi e Mero.",
        body,
    ))

    data = [
        ["Indicador", "4T2025", "4T2024", "Var. (%)"],
        ["Receita Liquida (R$ bi)", "142,7", "131,7", "+8,3%"],
        ["EBITDA Ajustado (R$ bi)", "68,4", "62,1", "+10,1%"],
        ["Lucro Liquido (R$ bi)", "31,2", "28,9", "+8,0%"],
        ["Producao (mil boed)", "2.820", "2.709", "+4,1%"],
        ["Divida Liquida (US$ bi)", "44,1", "47,3", "-6,8%"],
    ]
    tbl = Table(data, colWidths=[6 * cm, 3 * cm, 3 * cm, 3 * cm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#003366")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
    ]))
    flow.append(tbl)
    flow.append(Spacer(1, 0.4 * cm))

    flow.append(PageBreak())
    flow.append(Paragraph("<b>Investimentos (CAPEX)</b>", styles["Heading2"]))
    flow.append(Paragraph(
        "O CAPEX consolidado no 4T2025 foi de US$ 4,8 bilhoes, com 76% direcionado a "
        "Exploracao e Producao. Para o ciclo 2026-2030, o plano estrategico aprovado pelo "
        "Conselho prevê investimentos totais de US$ 111 bilhoes.",
        body,
    ))

    flow.append(Paragraph("<b>Dividendos e Remuneracao ao Acionista</b>", styles["Heading2"]))
    flow.append(Paragraph(
        "O Conselho de Administracao aprovou a distribuicao de dividendos no valor de "
        "R$ 1,75 por acao, totalizando R$ 22,8 bilhoes, com pagamento em duas parcelas: "
        "fevereiro e maio de 2026. A politica vigente preve distribuicao minima de 45% do "
        "fluxo de caixa livre quando a divida bruta estiver abaixo de US$ 65 bilhoes.",
        body,
    ))

    flow.append(Paragraph("<b>Endividamento</b>", styles["Heading2"]))
    flow.append(Paragraph(
        "A divida liquida encerrou o trimestre em US$ 44,1 bilhoes, reducao de 6,8% sobre "
        "o 4T2024. A relacao Divida Liquida/EBITDA ajustado foi de 0,87x, dentro do "
        "intervalo de conforto definido pela companhia.",
        body,
    ))

    flow.append(Paragraph("<b>Perspectivas para 2026</b>", styles["Heading2"]))
    flow.append(Paragraph(
        "A companhia projeta producao total entre 2,8 e 2,9 milhoes de boed para 2026, "
        "com entrada em operacao de duas novas plataformas no campo de Buzios. O CAPEX "
        "estimado e de US$ 21 bilhoes para o ano.",
        body,
    ))

    doc.build(flow)
    print(f"OK: {OUT / 'PETR4_Release_Resultados_4T2025.pdf'}")


def build_vale():
    doc = SimpleDocTemplate(
        str(OUT / "VALE3_Demonstracoes_Financeiras_2025.pdf"),
        pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm, topMargin=2 * cm, bottomMargin=2 * cm,
        title="VALE3 - Demonstracoes Financeiras 2025",
    )
    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("H1", parent=styles["Heading1"], textColor=colors.HexColor("#007a33"))
    body = styles["BodyText"]
    body.spaceAfter = 8
    flow = []

    flow.append(Paragraph("VALE S.A. (VALE3) - Demonstracoes Financeiras 2025", h1))
    flow.append(Paragraph("Relatorio Anual - Documento submetido a B3 e CVM", body))
    flow.append(Spacer(1, 0.4 * cm))

    flow.append(Paragraph("<b>Resumo Executivo</b>", styles["Heading2"]))
    flow.append(Paragraph(
        "A Vale registrou receita liquida de R$ 218,9 bilhoes no exercicio de 2025, "
        "praticamente estavel em relacao a 2024 (R$ 220,4 bilhoes). O EBITDA proforma "
        "atingiu R$ 92,1 bilhoes, com margem de 42,1%. O lucro liquido atribuivel aos "
        "acionistas foi de R$ 38,7 bilhoes.",
        body,
    ))

    flow.append(Paragraph("<b>Producao de Minerio de Ferro</b>", styles["Heading2"]))
    flow.append(Paragraph(
        "A producao de minerio de ferro alcancou 327 milhoes de toneladas em 2025, "
        "alta de 5,2% sobre 2024. Os embarques totalizaram 318 milhoes de toneladas, com "
        "preco medio realizado de US$ 98,4 por tonelada (CFR China, base 62%).",
        body,
    ))

    data = [
        ["Indicador", "2025", "2024"],
        ["Receita Liquida (R$ bi)", "218,9", "220,4"],
        ["EBITDA proforma (R$ bi)", "92,1", "94,8"],
        ["Lucro Liquido (R$ bi)", "38,7", "41,2"],
        ["Producao Minerio (Mt)", "327", "311"],
        ["CAPEX (US$ bi)", "6,1", "5,9"],
    ]
    tbl = Table(data, colWidths=[7 * cm, 4 * cm, 4 * cm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#007a33")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
    ]))
    flow.append(tbl)

    flow.append(PageBreak())
    flow.append(Paragraph("<b>Segmento de Metais Basicos</b>", styles["Heading2"]))
    flow.append(Paragraph(
        "A producao de cobre totalizou 348 mil toneladas em 2025 (+11% vs 2024), com "
        "destaque para o ramp-up da mina de Salobo III. A producao de niquel atingiu "
        "166 mil toneladas, alinhada ao guidance.",
        body,
    ))

    flow.append(Paragraph("<b>Politica de Dividendos</b>", styles["Heading2"]))
    flow.append(Paragraph(
        "O Conselho aprovou a distribuicao de US$ 4,2 bilhoes em dividendos referentes ao "
        "exercicio de 2025, equivalente a R$ 4,80 por acao. A politica de remuneracao "
        "mantem o piso de 30% do EBITDA ajustado menos investimentos correntes.",
        body,
    ))

    flow.append(Paragraph("<b>Acordo de Mariana e Brumadinho</b>", styles["Heading2"]))
    flow.append(Paragraph(
        "As provisoes consolidadas relativas aos acordos de reparacao de Mariana e "
        "Brumadinho totalizaram R$ 22,4 bilhoes ao final de 2025. Os pagamentos efetuados "
        "no exercicio somaram R$ 6,8 bilhoes.",
        body,
    ))

    doc.build(flow)
    print(f"OK: {OUT / 'VALE3_Demonstracoes_Financeiras_2025.pdf'}")


if __name__ == "__main__":
    build_petr4()
    build_vale()
