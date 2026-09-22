"""Generate marketplace chart PNGs at 700x380 to match other portfolio cases."""
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns

OUT = Path(__file__).resolve().parent / "images"
DPI = 100
FIGSIZE = (7.0, 3.8)  # 700x380 px at 100 dpi


def style():
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update(
        {
            "figure.figsize": FIGSIZE,
            "figure.dpi": DPI,
            "savefig.dpi": DPI,
            "font.size": 10,
            "axes.titlesize": 11,
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
    # Values from DataLens «Динамика продаж»
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
    color = "#4C78A8"
    ax.plot(months, values, marker="o", color=color, linewidth=2, markersize=5)
    ax.fill_between(months, values, alpha=0.12, color=color)
    ax.set_title("Динамика продаж")
    ax.set_ylabel("Выручка")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_money))
    ax.set_ylim(0, 1.45e9)
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(OUT / "sales_report.png", bbox_inches="tight", pad_inches=0.15)
    # Force exact canvas size: redraw at fixed size without tight crop drift
    plt.close(fig)

    # Re-save at exact 700x380 (tight_layout can change pixel size)
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    ax.plot(months, values, marker="o", color=color, linewidth=2, markersize=5)
    ax.fill_between(months, values, alpha=0.12, color=color)
    ax.set_title("Динамика продаж")
    ax.set_ylabel("Выручка")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_money))
    ax.set_ylim(0, 1.45e9)
    ax.tick_params(axis="x", rotation=25)
    fig.subplots_adjust(left=0.12, right=0.98, top=0.88, bottom=0.22)
    fig.savefig(OUT / "sales_report.png")
    plt.close(fig)


def category_managers():
    categories = [
        "Электроника",
        "Детские товары",
        "Строительство и ремонт",
        "Бытовая техника",
        "Компьютерная техника",
        "Товары для дома",
    ]
    revenue_m = np.array([2630.30, 627.55, 606.59, 585.03, 467.62, 195.95])  # млн ₽
    units = np.array([281_945, 1_204_064, 198_774, 151_509, 109_283, 350_026])

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE, dpi=DPI)
    palette = sns.color_palette("husl", n_colors=len(categories))

    # Left: revenue by category
    y = np.arange(len(categories))
    axes[0].barh(y, revenue_m, color=palette, height=0.7)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(categories, fontsize=8)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Факт, млн ₽")
    axes[0].set_title("Выручка по категориям")
    for i, v in enumerate(revenue_m):
        axes[0].text(v + 40, i, f"{v:.0f}", va="center", fontsize=8)

    # Right: plan vs fact by quarter
    quarters = ["апр '19", "июл '19", "окт '19"]
    fact = np.array([41, 521, 2395])
    plan = np.array([40, 467, 2511])
    x = np.arange(len(quarters))
    w = 0.35
    axes[1].bar(x - w / 2, fact, w, label="Факт", color="#4C78A8")
    axes[1].bar(x + w / 2, plan, w, label="План", color="#9ECAE1")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(quarters)
    axes[1].set_ylabel("млн ₽")
    axes[1].set_title("Факт и план по кварталам")
    axes[1].legend(fontsize=8, loc="upper left")

    axes[0].set_xlim(0, max(revenue_m) * 1.18)
    fig.subplots_adjust(left=0.28, right=0.98, top=0.88, bottom=0.18, wspace=0.4)
    fig.savefig(OUT / "category_managers.png")
    plt.close(fig)
    _ = units


def main():
    style()
    OUT.mkdir(parents=True, exist_ok=True)
    sales_dynamics()
    category_managers()
    for name in ("sales_report.png", "category_managers.png"):
        from PIL import Image

        im = Image.open(OUT / name)
        print(name, im.size)
        if im.size != (700, 380):
            im = im.resize((700, 380), Image.Resampling.LANCZOS)
            im.save(OUT / name)
            print("  resized ->", im.size)


if __name__ == "__main__":
    main()
