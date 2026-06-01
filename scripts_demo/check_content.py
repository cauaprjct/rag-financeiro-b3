"""Quick check: how much non-white content is in each screenshot?"""
from PIL import Image
from pathlib import Path

DOCS = Path(__file__).parent.parent / "docs"

for name in ["01-tela-inicial.png", "02-pdfs-selecionados.png", "03-indexado.png",
             "04-resposta-petrobras.png", "05-resposta-vale.png", "demo.png"]:
    p = DOCS / name
    if not p.exists():
        print(f"{name}: MISSING")
        continue
    img = Image.open(p).convert("RGB")
    # Count "dark" pixels (non-background)
    pixels = list(img.getdata())
    dark = sum(1 for r, g, b in pixels if r + g + b < 700)  # not near-white
    total = len(pixels)
    ratio = dark / total
    print(f"{name}: {img.size}  dark_ratio={ratio:.3%}")
