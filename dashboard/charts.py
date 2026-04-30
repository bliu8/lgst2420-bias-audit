"""
Chart helpers — each returns a matplotlib Figure.

Design goals:
- No chart titles (Streamlit markdown headers serve that role, avoiding overlap)
- Flagged groups are red, within-threshold groups are steel blue
- Reference lines show the mean so deviation is immediately visible
- Annotations state numbers plainly; layouts give enough margin to avoid overlap
"""
from __future__ import annotations

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

_BLUE  = "#4C72B0"
_RED   = "#C44E52"
_GRAY  = "#888888"
_BAND  = "#F0F0F0"

_LABEL_PAD = 0.03
_TOP_PAD   = 0.16


def _bar_colors(values: list[float], mean: float, threshold: float) -> list[str]:
    return [_RED if abs(v - mean) > threshold else _BLUE for v in values]


def _legend_handles(threshold: float) -> list[mpatches.Patch]:
    return [
        mpatches.Patch(color=_BLUE, label=f"Within ±{threshold:.0%} of mean"),
        mpatches.Patch(color=_RED,  label=f"Flagged  (>{threshold:.0%} from mean)"),
    ]


def positive_rate_chart(
    df: pd.DataFrame,
    mean_rate: float,
    threshold: float,
    scenario_label: str = "positive outcome",
) -> plt.Figure:
    values = df["positive_rate"].tolist()
    labels = [s.replace("_", " ") for s in df["subgroup"]]
    colors = _bar_colors(values, mean_rate, threshold)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    fig.subplots_adjust(left=0.14, right=0.97, top=0.93, bottom=0.12)

    bars = ax.bar(labels, values, color=colors, width=0.55, zorder=3)

    ax.axhline(mean_rate, color=_GRAY, linestyle="--", linewidth=1.2, zorder=2,
               label=f"Mean  {mean_rate:.1%}")
    ax.axhspan(max(0, mean_rate - threshold), min(1, mean_rate + threshold),
               color=_BAND, alpha=0.7, zorder=1)

    for bar, v, n in zip(bars, values, df["n"].tolist()):
        cx = bar.get_x() + bar.get_width() / 2
        # value label above bar
        ax.text(cx, v + _LABEL_PAD, f"{v:.0%}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
        # n= label inside bar (only if bar tall enough)
        if v > 0.12:
            ax.text(cx, v / 2, f"n={n}",
                    ha="center", va="center", fontsize=9,
                    color="white", fontweight="bold")

    top = min(1.0, max(values) + _LABEL_PAD + _TOP_PAD)
    ax.set_ylim(0, top)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_ylabel(f"% of responses giving a positive outcome", fontsize=9)
    ax.tick_params(axis="x", labelsize=10)
    ax.grid(axis="y", linestyle=":", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)

    handles = _legend_handles(threshold)
    handles.insert(0, mpatches.Patch(color=_GRAY, label=f"Mean  {mean_rate:.1%}",
                                     linestyle="--", fill=False))
    ax.legend(handles=_legend_handles(threshold) + [
        plt.Line2D([0], [0], color=_GRAY, linestyle="--", linewidth=1.2,
                   label=f"Mean  {mean_rate:.1%}")
    ], fontsize=8, loc="lower right")

    return fig


def deviation_chart(
    df: pd.DataFrame,
    mean_rate: float,
    threshold: float,
) -> plt.Figure:
    devs   = [v - mean_rate for v in df["positive_rate"].tolist()]
    labels = [s.replace("_", " ") for s in df["subgroup"]]
    colors = [_RED if abs(d) > threshold else _BLUE for d in devs]

    fig, ax = plt.subplots(figsize=(8, max(3.5, len(labels) * 0.85 + 0.8)))
    fig.subplots_adjust(left=0.18, right=0.88, top=0.95, bottom=0.14)

    bars = ax.barh(labels, devs, color=colors, height=0.5, zorder=3)

    ax.axvline(0, color=_GRAY, linewidth=1.2, zorder=2)
    ax.axvspan(-threshold, threshold, color=_BAND, alpha=0.8, zorder=1)

    x_max = max(abs(d) for d in devs) if devs else threshold
    ax.set_xlim(-(x_max + 0.12), x_max + 0.12)

    for bar, d in zip(bars, devs):
        # place label outside bar end, with enough margin to not clip
        offset = 0.015 if d >= 0 else -0.015
        ha     = "left"  if d >= 0 else "right"
        sign   = "+" if d >= 0 else ""
        ax.text(d + offset,
                bar.get_y() + bar.get_height() / 2,
                f"{sign}{d:.0%}",
                ha=ha, va="center", fontsize=9, fontweight="bold")

    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_xlabel("Deviation from mean positive rate", fontsize=9)
    ax.tick_params(axis="y", labelsize=10)
    ax.grid(axis="x", linestyle=":", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)

    ax.legend(handles=_legend_handles(threshold), fontsize=8,
              loc="lower right", bbox_to_anchor=(1.0, 0.0))
    return fig


def gap_summary_chart(gaps: dict, threshold: float) -> plt.Figure:
    """
    Horizontal bar chart of disparity gaps.
    Sentiment gap is shown on its own scale (−1 to 1) with a separate note,
    not as a percentage, because it can exceed 1.0 and displaying 140% is misleading.
    """
    entries = [
        ("positive_rate_gap", "Positive-rate gap\n(fraction saying 'yes')",          True),
        ("refusal_rate_gap",  "Refusal-rate gap\n(fraction refusing to answer)",      True),
        ("sentiment_gap",     "Sentiment gap\n(keyword tone score, scale −1 to +1)", False),
    ]

    names, values, is_pct, colors = [], [], [], []
    for key, label, pct in entries:
        v = gaps.get(key)
        if v is not None:
            names.append(label)
            values.append(v)
            is_pct.append(pct)
            # For sentiment (not percentage), compare raw value to threshold;
            # sentiment gap > 0.5 on a −1..+1 scale is a large effect
            flag_threshold = threshold if pct else 0.5
            colors.append(_RED if v > flag_threshold else _BLUE)

    if not values:
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.text(0.5, 0.5, "No gap data available", ha="center", va="center",
                transform=ax.transAxes, fontsize=11, color=_GRAY)
        ax.axis("off")
        return fig

    fig, ax = plt.subplots(figsize=(8, max(2.5, len(names) * 1.0 + 0.8)))
    fig.subplots_adjust(left=0.36, right=0.86, top=0.95, bottom=0.14)

    bars = ax.barh(names, values, color=colors, height=0.45, zorder=3)

    # Threshold line applies only to the percentage metrics
    ax.axvline(threshold, color=_RED, linestyle="--", linewidth=1.2, zorder=2,
               label=f"% threshold ({threshold:.0%})")

    for bar, v, pct in zip(bars, values, is_pct):
        label_str = f"{v:.1%}" if pct else f"{v:.2f}"
        ax.text(v + 0.008,
                bar.get_y() + bar.get_height() / 2,
                label_str,
                ha="left", va="center", fontsize=9, fontweight="bold")

    ax.set_xlim(0, max(values) + 0.18)
    # Mixed axis — show raw numbers, not forced percentage
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{x:.0%}" if x <= 1.5 else f"{x:.1f}"
    ))
    ax.set_xlabel("Gap magnitude  (% metrics: max group − min group)", fontsize=9)
    ax.tick_params(axis="y", labelsize=9)
    ax.grid(axis="x", linestyle=":", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)

    ax.legend(fontsize=8, loc="lower right")
    return fig


def refusal_chart(
    df: pd.DataFrame,
    mean_refusal: float,
    threshold: float,
) -> plt.Figure:
    values = df["refusal_rate"].tolist()
    labels = [s.replace("_", " ") for s in df["subgroup"]]
    colors = _bar_colors(values, mean_refusal, threshold)

    fig, ax = plt.subplots(figsize=(8, 4))
    fig.subplots_adjust(left=0.12, right=0.97, top=0.95, bottom=0.12)

    bars = ax.bar(labels, values, color=colors, width=0.55, zorder=3)

    ax.axhline(mean_refusal, color=_GRAY, linestyle="--", linewidth=1.2,
               label=f"Mean  {mean_refusal:.1%}")
    ax.axhspan(max(0, mean_refusal - threshold), min(1, mean_refusal + threshold),
               color=_BAND, alpha=0.7, zorder=1)

    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                v + _LABEL_PAD,
                f"{v:.0%}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")

    top = min(1.0, max(values) + _LABEL_PAD + _TOP_PAD) if values else 0.3
    ax.set_ylim(0, top)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_ylabel("% of responses that refused to answer", fontsize=9)
    ax.tick_params(axis="x", labelsize=10)
    ax.grid(axis="y", linestyle=":", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(handles=_legend_handles(threshold), fontsize=8, loc="upper right")
    return fig
