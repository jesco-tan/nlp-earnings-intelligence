"""Plotting helpers for NLP earnings intelligence, regional bank stress scenario."""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path

sns.set_theme(style="whitegrid", palette="muted")

_TIER_ORDER = ["gsib", "super_regional", "stressed"]
_TIER_LABELS = {"gsib": "GSIB", "super_regional": "Super-Regional", "stressed": "Stressed"}
_TIER_COLOURS = {"gsib": "#2196F3", "super_regional": "#4CAF50", "stressed": "#F44336"}
_FAILED = {"SIVB", "FRC"}


def plot_sentiment_heatmap(
    heatmap_df: pd.DataFrame,
    title: str = "Net Sentiment Score, Ticker × Quarter",
    save_path: Path | None = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(max(12, heatmap_df.shape[1] * 0.85), max(5, heatmap_df.shape[0] * 0.65)))
    sns.heatmap(
        heatmap_df.astype(float),
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        center=0,
        vmin=-0.5,
        vmax=0.5,
        ax=ax,
        cbar_kws={"label": "Mock net score (positive − negative)", "shrink": 0.8},
        linewidths=0.4,
    )
    ax.set_title(title, fontsize=13, pad=12)
    ax.set_xlabel("Quarter")
    ax.set_ylabel("Ticker")
    # Mark failed bank rows with a dagger
    ytick_labels = [t.get_text() for t in ax.get_yticklabels()]
    new_labels = [f"{t} †" if t in _FAILED else t for t in ytick_labels]
    ax.set_yticklabels(new_labels, fontsize=9)
    ax.text(
        0.01, -0.06, "† Programmed record cutoff", transform=ax.transAxes,
        fontsize=8, color="grey", style="italic",
    )
    plt.tight_layout()
    fig.text(0.5, -0.02, "Synthetic scenario; mock NLP outputs.", ha="center", fontsize=9)
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_sentiment_trends(
    quarterly_df: pd.DataFrame,
    anomaly_df: pd.DataFrame | None = None,
    save_path: Path | None = None,
) -> plt.Figure:
    tickers = quarterly_df["ticker"].unique()
    n = len(tickers)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.2, nrows * 3.2), sharex=False)
    axes = np.array(axes).flatten()

    # Colour by tier if available
    tier_map = {}
    if "tier" in quarterly_df.columns:
        tier_map = quarterly_df.drop_duplicates("ticker").set_index("ticker")["tier"].to_dict()

    for i, ticker in enumerate(tickers):
        ax = axes[i]
        sub = quarterly_df[quarterly_df["ticker"] == ticker].copy()
        qs = [str(q) for q in sub["year_quarter"]]
        colour = _TIER_COLOURS.get(tier_map.get(ticker, ""), "#607D8B")
        ax.plot(range(len(qs)), sub["avg_sentiment"], marker="o", color=colour,
                linewidth=2, markersize=4)

        if anomaly_df is not None:
            anom = anomaly_df[(anomaly_df["ticker"] == ticker) & (anomaly_df["anomaly"])]
            for _, row in anom.iterrows():
                idx_list = [j for j, q in enumerate(sub["year_quarter"].values)
                            if q == row["year_quarter"]]
                if idx_list:
                    ax.axvline(idx_list[0], color="red", linestyle="--",
                               linewidth=1.5, alpha=0.8)

        ax.set_title(f"{ticker}{' †' if ticker in _FAILED else ''}", fontsize=10,
                     color="darkred" if ticker in _FAILED else "black")
        ax.set_xticks(range(len(qs)))
        ax.set_xticklabels(qs, rotation=45, ha="right", fontsize=6)
        ax.axhline(0, color="grey", linewidth=0.8, linestyle=":")
        ax.set_ylabel("Sentiment", fontsize=8)
        ax.set_ylim(-0.65, 0.65)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Quarterly Net Sentiment Trends  (-- = anomaly flag)", fontsize=12, y=1.01)
    plt.tight_layout()
    fig.text(0.5, -0.02, "Synthetic scenario; mock NLP outputs.", ha="center", fontsize=9)
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_entity_freq(
    entity_df: pd.DataFrame,
    top_n: int = 15,
    save_path: Path | None = None,
) -> plt.Figure:
    top = entity_df.nlargest(top_n, "count")
    fig, ax = plt.subplots(figsize=(9, 5))
    colours = sns.color_palette("tab10", n_colors=top["label"].nunique())
    label_colours = dict(zip(top["label"].unique(), colours))
    bar_colours = [label_colours[l] for l in top["label"]]

    ax.barh(top["entity"], top["count"], color=bar_colours)
    ax.set_xlabel("Mention Count")
    ax.set_title(f"Top {top_n} Named Entities in SEC Filings")
    ax.invert_yaxis()

    patches = [mpatches.Patch(color=c, label=l) for l, c in label_colours.items()]
    ax.legend(handles=patches, loc="lower right", fontsize=8)
    plt.tight_layout()
    fig.text(0.5, -0.02, "Synthetic scenario; mock NLP outputs.", ha="center", fontsize=9)
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_section_gap(
    section_df: pd.DataFrame,
    tickers_highlight: list[str] | None = None,
    gap_alert: float = 0.60,
    save_path: Path | None = None,
) -> plt.Figure:
    """
    Plot MD&A sentiment vs risk_factors sentiment over time for each ticker,
    shading the region between them to show the section gap.
    """
    tickers = section_df["ticker"].unique()
    n = len(tickers)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.2, nrows * 3.2), sharex=False)
    axes = np.array(axes).flatten()

    tier_map = {}
    if "tier" in section_df.columns:
        tier_map = section_df.drop_duplicates("ticker").set_index("ticker")["tier"].to_dict()

    for i, ticker in enumerate(tickers):
        ax = axes[i]
        sub = section_df[section_df["ticker"] == ticker].copy().sort_values("year_quarter")
        qs = list(range(len(sub)))

        if "mda_sentiment" in sub.columns and "risk_sentiment" in sub.columns:
            mda = sub["mda_sentiment"].values
            risk = sub["risk_sentiment"].values
            ax.plot(qs, mda, marker="o", color="#2196F3", linewidth=2,
                    markersize=4, label="MD&A")
            ax.plot(qs, risk, marker="s", color="#F44336", linewidth=2,
                    markersize=4, label="Risk Factors")
            ax.fill_between(qs, risk, mda, alpha=0.15, color="#FF9800")

            # Flag quarters where gap exceeds alert threshold
            if "section_gap" in sub.columns:
                alert_mask = sub["section_gap"].values > gap_alert
                alert_xs = [x for x, m in zip(qs, alert_mask) if m]
                if alert_xs:
                    for ax_x in alert_xs:
                        ax.axvline(ax_x, color="orange", linestyle=":",
                                   linewidth=1.5, alpha=0.9)

        q_labels = [str(q) for q in sub["year_quarter"]]
        ax.set_xticks(qs)
        ax.set_xticklabels(q_labels, rotation=45, ha="right", fontsize=6)
        ax.axhline(0, color="grey", linewidth=0.6, linestyle=":")
        ax.set_ylim(-0.85, 0.75)
        ax.set_title(f"{ticker}{' †' if ticker in _FAILED else ''}",
                     fontsize=10, color="darkred" if ticker in _FAILED else "black")
        ax.set_ylabel("Sentiment", fontsize=8)
        if i == 0:
            ax.legend(fontsize=7, loc="upper right")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(
        "MD&A vs Risk-Factors Sentiment  (shaded = gap, vertical lines = gap-exceedance quarters)",
        fontsize=12, y=1.01,
    )
    plt.tight_layout()
    fig.text(0.5, -0.02, "Synthetic scenario; mock NLP outputs.", ha="center", fontsize=9)
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_tier_comparison(
    tier_agg: pd.DataFrame,
    save_path: Path | None = None,
) -> plt.Figure:
    """
    Line chart comparing quarterly sentiment across GSIB / super-regional / stressed tiers.
    """
    fig, ax = plt.subplots(figsize=(13, 5))

    for tier in _TIER_ORDER:
        sub = tier_agg[tier_agg["tier"] == tier].sort_values("year_quarter")
        if sub.empty:
            continue
        xs = list(range(len(sub)))
        colour = _TIER_COLOURS[tier]
        label = _TIER_LABELS[tier]
        ax.plot(xs, sub["avg_sentiment"], marker="o", color=colour,
                linewidth=2.5, markersize=5, label=label)

    # x-axis labels from first tier present
    ref = tier_agg[tier_agg["tier"] == _TIER_ORDER[0]].sort_values("year_quarter")
    if not ref.empty:
        labels = [str(q) for q in ref["year_quarter"]]
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)

    ax.axhline(0, color="grey", linewidth=0.8, linestyle=":")
    ax.axvspan(4, 7, alpha=0.05, color="red", label="COVID stress (2020 Q2–Q4)")
    ax.axvspan(8, 13, alpha=0.05, color="orange", label="Rate hiking (2022 Q1 – 2023 Q2)")
    ax.set_ylabel("Average Net Sentiment")
    ax.set_title("Quarterly Sentiment by Bank Tier, GSIB vs Super-Regional vs Stressed", fontsize=12)
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.text(0.5, -0.02, "Synthetic scenario; mock NLP outputs.", ha="center", fontsize=9)
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_score_distribution(
    sentiment_df: pd.DataFrame,
    save_path: Path | None = None,
) -> plt.Figure:
    """
    Histogram/KDE of net sentiment scores split by section (MD&A vs risk factors)
    and by tier, shows structural differences in sentiment distributions.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Panel 1: by section
    ax = axes[0]
    for section, colour in [("mda", "#2196F3"), ("risk_factors", "#F44336")]:
        sub = sentiment_df[sentiment_df["section"] == section]["score"].dropna()
        ax.hist(sub, bins=40, alpha=0.55, color=colour,
                label=f"{'MD&A' if section == 'mda' else 'Risk Factors'} (n={len(sub)})",
                density=True)
    ax.axvline(0, color="black", linewidth=0.9, linestyle="--")
    ax.set_xlabel("Net Sentiment Score")
    ax.set_title("Distribution by Section Type")
    ax.legend(fontsize=9)

    # Panel 2: by tier
    ax2 = axes[1]
    if "tier" in sentiment_df.columns:
        for tier in _TIER_ORDER:
            sub = sentiment_df[sentiment_df["tier"] == tier]["score"].dropna()
            colour = _TIER_COLOURS[tier]
            ax2.hist(sub, bins=35, alpha=0.55, color=colour,
                     label=f"{_TIER_LABELS[tier]} (n={len(sub)})",
                     density=True)
    else:
        ax2.text(0.5, 0.5, "No tier column available", ha="center", va="center",
                 transform=ax2.transAxes)
    ax2.axvline(0, color="black", linewidth=0.9, linestyle="--")
    ax2.set_xlabel("Net Sentiment Score")
    ax2.set_title("Distribution by Bank Tier")
    ax2.legend(fontsize=9)

    plt.suptitle("Net Sentiment Score Distributions", fontsize=12)
    plt.tight_layout()
    fig.text(0.5, -0.02, "Synthetic scenario; mock NLP outputs.", ha="center", fontsize=9)
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_survivorship(
    surv_df: pd.DataFrame,
    save_path: Path | None = None,
) -> plt.Figure:
    """
    Dual-line chart showing sector average sentiment with vs without failed banks.
    Compares available-record and retrospectively selected survivor cohorts; it does not attribute later recovery.
    is driven by SIVB and FRC dropping out of the corpus.
    """
    fig, ax = plt.subplots(figsize=(13, 5))

    qs = list(range(len(surv_df)))
    ax.plot(qs, surv_df["avg_sentiment_all"], marker="o", color="#607D8B",
            linewidth=2, markersize=4, label="All banks (incl. SIVB & FRC)")
    ax.plot(qs, surv_df["avg_sentiment_survivors"], marker="^", color="#4CAF50",
            linewidth=2, markersize=5, linestyle="--",
            label="Survivors only (SIVB & FRC excluded)")

    ax.fill_between(qs,
                    surv_df["avg_sentiment_all"],
                    surv_df["avg_sentiment_survivors"],
                    alpha=0.2, color="#FF9800",
                    label="Retrospective cohort difference")

    labels = [str(q) for q in surv_df["year_quarter"]]
    ax.set_xticks(qs)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.axhline(0, color="grey", linewidth=0.8, linestyle=":")

    # Mark period where failed banks exit
    exit_idx = next((i for i, q in enumerate(surv_df["year_quarter"])
                     if str(q) >= "2023Q1"), None)
    if exit_idx is not None:
        ax.axvline(exit_idx, color="red", linewidth=1.5, linestyle=":",
                   label="Programmed record cutoff (2023 Q1)")

    ax.set_ylabel("Average Net Sentiment (all sections)")
    ax.set_title(
        "Retrospective mock-score cohort comparison",
        fontsize=12,
    )
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.text(0.5, -0.02, "Synthetic scenario; mock NLP outputs.", ha="center", fontsize=9)
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
