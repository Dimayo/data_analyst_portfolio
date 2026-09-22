"""Generate marketplace chart PNGs at 700x380 — seaborn darkgrid (white grids), like sporting_store ROC."""
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns
from PIL import Image

OUT = Path(__file__).resolve().parent / "images"
DPI = 100
FIGSIZE = (7.0, 3.8)  # 700x380
TARGET = (700, 380)
FIG_BG = "white"
BLUE = "#4C78A8"
BLUE_LIGHT = "#9ECAE1"


def style():
    # Same as sporting_store: default seaborn theme → darkgrid, white grid lines
    sns.set_theme()
    plt.rcParams.update(
        {
            "figure.figsize": FIGSIZE,
            "figure.dpi": DPI,
            "savefig.dpi": DPI,
            "figure.facecolor": FIG_BG,
            "savefig.facecolor": FIG_BG,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "medium",
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )


def fmt_money(x, _pos=None):
    if abs(x) >= 1e9:
        return f"{x / 1e9:.2f}B"
    if abs(x) >= 1e6:
        v = x / 1e6
        return f"{v:.0f}M" if v >= 10 else f"{v:.1f}M"
    return f"{x:,.0f}"


def trim_edges(im: Image.Image, keep_pad: int = 4) -> Image.Image:
    """Crop excess white on top/bottom/right; keep left padding for y-label."""
    arr = np.asarray(im.convert("RGB"))
    mask = (arr < 248).any(axis=2)
    ys = np.where(mask.any(axis=1))[0]
    xs = np.where(mask.any(axis=0))[0]
    if len(ys) == 0 or len(xs) == 0:
        return im
    top = max(0, int(ys.min()) - keep_pad)
    bottom = min(im.size[1], int(ys.max()) + 1 + keep_pad)
    left = max(0, int(xs.min()) - keep_pad)
    right = min(im.size[0], int(xs.max()) + 1 + keep_pad)
    cropped = im.crop((left, top, right, bottom))
    return cropped.resize(TARGET, Image.Resampling.LANCZOS)


def save_chart(fig, path: Path):
    # Left: snug to y-label; bottom tighter for rotated tick labels
    fig.subplots_adjust(left=0.10, right=0.99, top=0.93, bottom=0.16)
    fig.savefig(path, dpi=DPI, facecolor=FIG_BG)
    plt.close(fig)
    im = Image.open(path)
    if im.size != TARGET:
        im = im.resize(TARGET, Image.Resampling.LANCZOS)
    im = trim_edges(im, keep_pad=4)
    im.save(path)


def sales_dynamics():
    months = [
        "апр '19",
        "май '19",
        "июн '19",
        "июл '19",
        "авг '19",
        "сен '19",
        "окт '19",
        "ноя '19",
        "дек '19",
    ]
    values = np.array(
        [
            1.38e6,
            16.29e6,
            22.75e6,
            52.91e6,
            122.89e6,
            305.74e6,
            423.41e6,
            696.30e6,
            1.27e9,
        ]
    )

    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    fig.patch.set_facecolor(FIG_BG)
    ax.plot(months, values, marker="o", color=BLUE, linewidth=2.2, markersize=5.5)
    ax.fill_between(np.arange(len(months)), values, alpha=0.15, color=BLUE)
    ax.set_xticks(np.arange(len(months)))
    ax.set_xticklabels(months)
    ax.set_xlim(-0.35, len(months) - 0.65)
    ax.set_title("Динамика продаж", fontsize=12, pad=4)
    ax.set_ylabel("Выручка", labelpad=6)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_money))
    ax.set_ylim(0, 1.45e9)
    ax.tick_params(axis="x", rotation=25)
    ax.set_axisbelow(True)
    save_chart(fig, OUT / "sales_dynamics.png")


def category_managers():
    quarters = ["апр '19", "июл '19", "окт '19"]
    fact = np.array([41, 521, 2395])
    plan = np.array([40, 467, 2511])
    x = np.arange(len(quarters))
    w = 0.36

    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    fig.patch.set_facecolor(FIG_BG)
    ax.bar(x - w / 2, fact, w, label="Факт", color=BLUE, zorder=3)
    ax.bar(x + w / 2, plan, w, label="План", color=BLUE_LIGHT, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(quarters)
    ax.set_ylabel("млн ₽", labelpad=6)
    ax.set_title("Факт и план по кварталам", fontsize=12, pad=4)
    ax.legend(fontsize=9, loc="upper left")
    ax.set_axisbelow(True)
    save_chart(fig, OUT / "plan_vs_fact.png")


def main():
    style()
    OUT.mkdir(parents=True, exist_ok=True)
    sales_dynamics()
    category_managers()
    for name in ("sales_dynamics.png", "plan_vs_fact.png"):
        im = Image.open(OUT / name)
        print(name, im.size)


if __name__ == "__main__":
    main()
