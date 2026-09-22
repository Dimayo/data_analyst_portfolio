"""Pad LTV heatmap to 700x380 without redrawing titles (keep matplotlib look)."""
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent / "images"
TARGET = (700, 380)
BG = (255, 255, 255)


def pad_to(path: Path, size=TARGET):
    im = Image.open(path).convert("RGB")
    if im.size == size:
        print(path.name, "already", size)
        return
    canvas = Image.new("RGB", size, BG)
    ox = (size[0] - im.size[0]) // 2
    oy = (size[1] - im.size[1]) // 2
    canvas.paste(im, (ox, oy))
    canvas.save(path)
    print(path.name, im.size, "->", size, "pad", (ox, oy))


def main():
    pad_to(ROOT / "source_revenue_heatmap.png")
    # cumulative already 700x380 with original title
    im = Image.open(ROOT / "cumulative_ltv.png")
    print("cumulative_ltv.png", im.size)


if __name__ == "__main__":
    main()
