"""Print top-of-image OCR-like text by inspecting pixels (just dimensions check)."""
from PIL import Image
from pathlib import Path

ROOT = Path(__file__).parent.parent / "docs"

for p in sorted(ROOT.glob("*.png")):
    if p.name.startswith("_"):
        continue
    img = Image.open(p)
    print(f"{p.name}: {img.size}, mode={img.mode}")
    # Extract average color of top 50 px to verify it's not blank
    top = img.crop((0, 0, img.width, 50)).convert("RGB")
    pixels = list(top.getdata())
    avg = (
        sum(p[0] for p in pixels) // len(pixels),
        sum(p[1] for p in pixels) // len(pixels),
        sum(p[2] for p in pixels) // len(pixels),
    )
    print(f"  top avg RGB: {avg}")
