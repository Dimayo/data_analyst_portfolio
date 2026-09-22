"""Fit LTV charts to 700x380 and match title size on cumulative_ltv to heatmap."""
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent / "images"
TARGET = (700, 380)
PAD = 2
BG = (255, 255, 255)


def content_box(im: Image.Image, thr=250):
    arr = np.asarray(im.convert("RGB"))
    mask = (arr < thr).any(axis=2)
    ys, xs = np.where(mask)
    return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)


def fit_canvas(im: Image.Image, size=TARGET, pad=PAD, bg=BG) -> Image.Image:
    box = content_box(im)
    cropped = im.crop(box).convert("RGB")
    tw, th = size[0] - 2 * pad, size[1] - 2 * pad
    cw, ch = cropped.size
    scale = min(tw / cw, th / ch)
    nw, nh = max(1, int(round(cw * scale))), max(1, int(round(ch * scale)))
    resized = cropped.resize((nw, nh), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, bg)
    canvas.paste(resized, ((size[0] - nw) // 2, (size[1] - nh) // 2))
    return canvas


def load_font(size: int) -> ImageFont.ImageFont:
    for name in (
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/calibri.ttf",
    ):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def title_ink_height(im: Image.Image, max_y=50) -> int:
    arr = np.asarray(im.convert("RGB"))
    dark = (arr < 60).any(axis=2)
    rows = np.where(dark.any(axis=1) & (np.arange(arr.shape[0]) < max_y))[0]
    if len(rows) == 0:
        return 0
    return int(rows.max() - rows.min() + 1)


def replace_title(im: Image.Image, text: str, font_size: int, band=36) -> Image.Image:
    out = im.convert("RGB").copy()
    w, _ = out.size
    draw = ImageDraw.Draw(out)
    # Sample background from a title-side pixel (usually white)
    draw.rectangle([0, 0, w, band], fill=BG)
    font = load_font(font_size)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (w - tw) // 2
    y = max(2, (band - th) // 2 - 1)
    draw.text((x, y), text, fill=(0, 0, 0), font=font)
    return out


def main():
    heatmap_path = ROOT / "source_revenue_heatmap.png"
    ltv_path = ROOT / "cumulative_ltv.png"

    # Heatmap: only normalize canvas size (keep existing title)
    hm = fit_canvas(Image.open(heatmap_path))
    hm.save(heatmap_path)

    # Cumulative: normalize + redraw title to match heatmap weight
    ref_h = title_ink_height(hm)
    # Heatmap title ink ≈ 12–16px after fit; use 15pt for parity
    font_size = 15
    ltv = fit_canvas(Image.open(ltv_path))
    ltv = replace_title(ltv, "Накопительный LTV по когортам", font_size)
    ltv.save(ltv_path)

    for p in (heatmap_path, ltv_path, ROOT / "avg_cheque.png"):
        im = Image.open(p)
        box = content_box(im)
        pads = (box[0], box[1], im.size[0] - box[2], im.size[1] - box[3])
        print(p.name, im.size, "pads", pads, "title_h", title_ink_height(im), "ref", ref_h)


if __name__ == "__main__":
    main()
