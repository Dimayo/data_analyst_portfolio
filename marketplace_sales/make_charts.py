"""Generate marketplace chart PNGs at 700x380 — light grey seaborn style, one chart each."""
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns

OUT = Path(__file__).resolve().parent / "images"
DPI = 100
FIGSIZE = (7.0, 3.8)  # 700x380 px at 100 dpi

BLUE = "#4C78A8"
BLUE_LIGHT = "#9ECAE1"


def style():
    # Same as other cases: white figure, light-grey axes (seaborn whitegrid)
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update(
        {
            "figure.figsize": FIGSIZE,
            "figure.dpi": DPI,
            "savefig.dpi": DPI,
            "figure.facecolor": "white",
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
    ax.plot(months, values, marker="o", color=BLUE, linewidth=2.2, markersize=5.5)
    ax.fill_between(months, values, alpha=0.15, color=BLUE)
    ax.set_title("Динамика продаж")
    ax.set_ylabel("Выручка")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_money))
    ax.set_ylim(0, 1.45e9)
    ax.tick_params(axis="x", rotation=25)
    ax.set_axisbelow(True)
    fig.subplots_adjust(left=0.12, right=0.98, top=0.88, bottom=0.22)
    fig.savefig(OUT / "sales_report.png")
    plt.close(fig)


def category_managers():
    quarters = ["апр '19", "июл '19", "окт '19"]
    fact = np.array([41, 521, 2395])
    plan = np.array([40, 467, 2511])
    x = np.arange(len(quarters))
    w = 0.36

    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    ax.bar(x - w / 2, fact, w, label="Факт", color=BLUE, zorder=3)
    ax.bar(x + w / 2, plan, w, label="План", color=BLUE_LIGHT, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(quarters)
    ax.set_ylabel("млн ₽")
    ax.set_title("Факт и план по кварталам")
    ax.legend(fontsize=9, loc="upper left")
    ax.set_axisbelow(True)
    fig.subplots_adjust(left=0.12, right=0.98, top=0.88, bottom=0.16)
    fig.savefig(OUT / "category_managers.png")
    plt.close(fig)


def main():
    style()
    OUT.mkdir(parents=True, exist_ok=True)
    sales_dynamics()
    category_managers()
    from PIL import Image

    for name in ("sales_report.png", "category_managers.png"):
        im = Image.open(OUT / name)
        print(name, im.size)
        if im.size != (700, 380):
            im = im.resize((700, 380), Image.Resampling.LANCZOS)
            im.save(OUT / name)
            print("  resized ->", im.size)


if __name__ == "__main__":
    main()
