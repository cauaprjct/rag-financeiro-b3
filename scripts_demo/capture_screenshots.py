"""Captura screenshots da aplicacao Streamlit usando Playwright sem auto-traducao.

Pre-requisito: o Streamlit ja deve estar rodando em http://localhost:8501.
Usar: .venv\\Scripts\\python.exe scripts_demo\\capture_screenshots.py
"""

import asyncio
from pathlib import Path

from playwright.async_api import async_playwright

ROOT = Path(__file__).parent.parent
DOCS = ROOT / "docs"
DOCS.mkdir(exist_ok=True)
PDFS = [
    str(ROOT / "data" / "sample_docs" / "PETR4_Release_Resultados_4T2025.pdf"),
    str(ROOT / "data" / "sample_docs" / "VALE3_Demonstracoes_Financeiras_2025.pdf"),
]


async def shot(page, name: str, full: bool = True):
    out = DOCS / name
    await page.wait_for_timeout(500)
    if full:
        # Streamlit puts content inside <section class="stMain"> with its own
        # scroll. Resize viewport so the section grows to its full content.
        info = await page.evaluate(
            "() => { const m = document.querySelector('section.stMain'); "
            "return m ? { sh: m.scrollHeight, ch: m.clientHeight } : null; }"
        )
        print(f"  before: section sh={info and info.get('sh')} ch={info and info.get('ch')}")
        if info and info["sh"] > info["ch"]:
            target_h = max(900, info["sh"] + 60)
            await page.set_viewport_size({"width": 1440, "height": target_h})
            await page.wait_for_timeout(600)
            info2 = await page.evaluate(
                "() => { const m = document.querySelector('section.stMain'); "
                "return m ? { sh: m.scrollHeight, ch: m.clientHeight } : null; }"
            )
            print(f"  after resize to h={target_h}: sh={info2 and info2.get('sh')} ch={info2 and info2.get('ch')}")
    await page.screenshot(path=str(out), full_page=full)
    if full:
        await page.set_viewport_size({"width": 1440, "height": 900})
        await page.wait_for_timeout(150)
    print(f"Salvo: docs/{name}")


async def main():
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(
            headless=True,
            args=[
                "--disable-features=Translate,TranslateUI,AutoTranslate",
                "--disable-translate",
                "--lang=pt-BR",
            ],
        )
        context = await browser.new_context(
            viewport={"width": 1440, "height": 900},
            locale="pt-BR",
            extra_http_headers={"Accept-Language": "pt-BR,pt;q=0.9,en;q=0.5"},
        )
        page = await context.new_page()

        await page.goto("http://localhost:8501", wait_until="networkidle")
        await page.wait_for_selector("text=Sua pergunta", timeout=30000)
        await page.wait_for_timeout(1500)

        translated = await page.evaluate(
            "() => Array.from(document.querySelectorAll('button')).some(b => b.querySelector('font'))"
        )
        print(f"DOM traduzido pelo browser? {translated}")

        # 1) Limpa documentos previamente indexados (Chroma e persistente)
        trash_buttons = await page.query_selector_all("button:has-text('🗑')")
        for tb in trash_buttons:
            try:
                await tb.click()
                await page.wait_for_timeout(800)
            except Exception:
                pass
        await page.wait_for_timeout(1500)

        # 1) Tela inicial (vazia)
        await shot(page, "01-tela-inicial.png")

        # 2) PDFs selecionados (antes de indexar)
        file_input = await page.query_selector("input[type='file']")
        await file_input.set_input_files(PDFS)
        await page.wait_for_timeout(2000)
        await shot(page, "02-pdfs-selecionados.png")

        # 3) Indexado
        indexar = page.locator("button", has_text="Indexar")
        await indexar.click()
        await page.wait_for_selector("text=indexado", timeout=120000)
        await page.wait_for_timeout(2000)
        await shot(page, "03-indexado.png")

        # 4) Pergunta sobre Petrobras + resposta com citacoes
        textarea = page.locator("input[aria-label='Sua pergunta']").first
        await textarea.fill(
            "Qual foi o lucro liquido da Petrobras no 4T2025 e qual o valor "
            "dos dividendos aprovados?"
        )
        perguntar = page.locator("button", has_text="Perguntar")
        await perguntar.click()
        await page.wait_for_selector(
            "text=PETR4_Release_Resultados_4T2025.pdf", timeout=90000, state="visible"
        )
        await page.wait_for_timeout(2500)
        await shot(page, "04-resposta-petrobras.png")

        # Hero/demo screenshot for README - viewport size, scrolled to top
        await page.evaluate("() => window.scrollTo(0, 0)")
        await page.wait_for_timeout(500)
        await shot(page, "demo.png", full=False)

        # 5) Pergunta sobre Vale
        await textarea.fill("Qual foi a producao de minerio de ferro da Vale em 2025?")
        await perguntar.click()
        await page.wait_for_selector(
            "text=VALE3_Demonstracoes_Financeiras_2025.pdf",
            timeout=90000,
            state="visible",
        )
        await page.wait_for_timeout(2500)
        await shot(page, "05-resposta-vale.png")

        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
