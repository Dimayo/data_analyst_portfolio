"""Fix LTV heatmaps: drop black top strip, pad to 700x380, align left edges."""
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent / "images"
TARGET = (700, 380)
BG = (255, 255, 255)
LEFT_PAD = 2
TOP_PAD = 2


def drop_dark_top(im: Image.Image, thr: int = 40) -> Image.Image:
    arr = np.asarray(im.convert("RGB"))
    # Drop leading rows that are almost entirely very dark (artifact strip)
    while arr.shape[0] > 1 and (arr[0].min(axis=1) < thr).mean() > 0.5:
        arr = arr[1:]
    return Image.fromarray(arr)


def content_box(arr: np.ndarray, thr: int = 250):
    mask = (arr < thr).any(axis=2)
    ys, xs = np.where(mask)
    return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)


def place_on_canvas(im: Image.Image, size=TARGET) -> Image.Image:
    """Crop to ink, scale to fit, paste with fixed left/top pads (aligned pair)."""
    arr = np.asarray(im.convert("RGB"))
    x0, y0, x1, y1 = content_box(arr)
    cropped = im.convert("RGB").crop((x0, y0, x1, y1))
    tw = size[0] - LEFT_PAD - 2
    th = size[1] - TOP_PAD - 2
    cw, ch = cropped.size
    scale = min(tw / cw, th / ch)
    nw, nh = max(1, int(round(cw * scale))), max(1, int(round(ch * scale)))
    resized = cropped.resize((nw, nh), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, BG)
    # Left-align both charts so y-labels line up in the README
    canvas.paste(resized, (LEFT_PAD, TOP_PAD))
    return canvas


def main():
    hm_path = ROOT / "source_revenue_heatmap.png"
    ltv_path = ROOT / "cumulative_ltv.png"

    hm = drop_dark_top(Image.open(hm_path))
    ltv = Image.open(ltv_path)

    hm2 = place_on_canvas(hm)
    ltv2 = place_on_canvas(ltv)
    hm2.save(hm_path)
    ltv2.save(ltv_path)

    for p in (hm_path, ltv_path):
        im = Image.open(p)
        a = np.asarray(im)
        print(p.name, im.size, "top_mean", a[0].mean(axis=0).astype(int).tolist(), "left_ink", int(np.where((a < 250).any(axis=2).any(axis=0))[0].min()))


if __name__ == "__main__":
    main()
