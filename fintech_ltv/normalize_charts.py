"""Pad/crop LTV charts to 700x380; unify title font without clipping data rows."""
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent / "images"
TARGET = (700, 380)
BG = (255, 255, 255)


def load_font(size: int) -> ImageFont.ImageFont:
    for name in (
        r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\segoeui.ttf",
        r"C:\Windows\Fonts\calibri.ttf",
    ):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def content_box(arr: np.ndarray, thr=250):
    mask = (arr < thr).any(axis=2)
    ys, xs = np.where(mask)
    return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)


def fit_canvas(im: Image.Image, size=TARGET, pad=2) -> Image.Image:
    arr = np.asarray(im.convert("RGB"))
    x0, y0, x1, y1 = content_box(arr)
    cropped = im.convert("RGB").crop((x0, y0, x1, y1))
    tw, th = size[0] - 2 * pad, size[1] - 2 * pad
    cw, ch = cropped.size
    scale = min(tw / cw, th / ch)
    nw, nh = max(1, int(round(cw * scale))), max(1, int(round(ch * scale)))
    resized = cropped.resize((nw, nh), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, BG)
    canvas.paste(resized, ((size[0] - nw) // 2, (size[1] - nh) // 2))
    return canvas


def data_start_y(arr: np.ndarray) -> int:
    """First row that looks like heatmap cells (many non-white colored pixels)."""
    h, w, _ = arr.shape
    for y in range(h):
        row = arr[y]
        # saturated / non-gray color pixels
        mx = row.max(axis=1)
        mn = row.min(axis=1)
        colorful = ((mx - mn) > 25) & (mx < 250)
        if colorful.mean() > 0.08:
            return y
    return 40


def set_title(im: Image.Image, text: str, font_size: int = 13) -> Image.Image:
    out = im.convert("RGB")
    arr = np.asarray(out)
    start = data_start_y(arr)
    # Leave a few px gap above data
    band = max(28, min(start - 4, 48))
    draw = ImageDraw.Draw(out)
    draw.rectangle([0, 0, out.size[0], band], fill=BG)
    font = load_font(font_size)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    # If title doesn't fit, shrink font
    while tw > out.size[0] - 16 and font_size > 10:
        font_size -= 1
        font = load_font(font_size)
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (out.size[0] - tw) // 2
    y = max(2, (band - th) // 2)
    draw.text((x, y), text, fill=(0, 0, 0), font=font)
    return out


def main():
    hm_path = ROOT / "source_revenue_heatmap.png"
    ltv_path = ROOT / "cumulative_ltv.png"

    # Same canvas + same title font size for both
    font_size = 13
    hm = set_title(fit_canvas(Image.open(hm_path)), "Выручка по месяцам и каналам", font_size)
    ltv = set_title(fit_canvas(Image.open(ltv_path)), "Накопительный LTV по когортам", font_size)
    hm.save(hm_path)
    ltv.save(ltv_path)

    for p, im in ((hm_path, hm), (ltv_path, ltv)):
        print(p.name, im.size)


if __name__ == "__main__":
    main()
