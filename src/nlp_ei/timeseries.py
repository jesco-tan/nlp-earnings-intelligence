"""Quarterly sentiment aggregation, anomaly detection, and disaggregated analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd


def aggregate_quarterly(
    sentiment_df: pd.DataFrame,
    ticker_col: str = "ticker",
    date_col: str = "filing_date",
    score_col: str = "score",
) -> pd.DataFrame:
    """Aggregate per-document sentiment to quarterly ticker-level series (both sections combined)."""
    df = sentiment_df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df["year_quarter"] = df[date_col].dt.to_period("Q")

    agg = (
        df.groupby([ticker_col, "year_quarter"])[score_col]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "avg_sentiment", "std": "sentiment_std", "count": "n_docs"})
    )
    return agg.sort_values([ticker_col, "year_quarter"])


def aggregate_quarterly_by_section(
    sentiment_df: pd.DataFrame,
    ticker_col: str = "ticker",
    date_col: str = "filing_date",
    section_col: str = "section",
    score_col: str = "score",
) -> pd.DataFrame:
    """
    Aggregate sentiment separately for each section (mda vs risk_factors) per quarter.

    Returns a wide DataFrame with columns:
      ticker, year_quarter, mda_sentiment, risk_sentiment, n_mda, n_risk
    """
    df = sentiment_df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df["year_quarter"] = df[date_col].dt.to_period("Q")

    long = (
        df.groupby([ticker_col, "year_quarter", section_col])[score_col]
        .mean()
        .reset_index()
        .rename(columns={score_col: "avg_sentiment"})
    )
    counts = (
        df.groupby([ticker_col, "year_quarter", section_col])[score_col]
        .count()
        .reset_index()
        .rename(columns={score_col: "n_docs"})
    )

    pivot = long.pivot_table(
        index=[ticker_col, "year_quarter"],
        columns=section_col,
        values="avg_sentiment",
    ).reset_index()
    pivot.columns.name = None

    cnt_pivot = counts.pivot_table(
        index=[ticker_col, "year_quarter"],
        columns=section_col,
        values="n_docs",
        aggfunc="sum",
    ).reset_index()
    cnt_pivot.columns.name = None

    for col, alias in [("mda", "n_mda"), ("risk_factors", "n_risk")]:
        if col in cnt_pivot.columns:
            pivot[alias] = cnt_pivot[col].values

    col_map = {}
    if "mda" in pivot.columns:
        col_map["mda"] = "mda_sentiment"
    if "risk_factors" in pivot.columns:
        col_map["risk_factors"] = "risk_sentiment"
    pivot = pivot.rename(columns=col_map)

    return pivot.sort_values([ticker_col, "year_quarter"]).reset_index(drop=True)


def compute_section_gap(
    section_df: pd.DataFrame,
    ticker_col: str = "ticker",
    mda_col: str = "mda_sentiment",
    risk_col: str = "risk_sentiment",
) -> pd.DataFrame:
    """
    Compute the MD&A–risk-factors sentiment gap per (ticker, quarter).

    A large positive gap means management is telling an optimistic operating story
    in MD&A while simultaneously flagging severe risks in disclosures, an
    early-warning sign of optimism bias rather than balance-sheet strength.
    """
    df = section_df.copy()
    if mda_col in df.columns and risk_col in df.columns:
        df["section_gap"] = df[mda_col] - df[risk_col]
    else:
        df["section_gap"] = np.nan

    # Z-score the gap within ticker to highlight unusual widening
    df["gap_zscore"] = df.groupby(ticker_col)["section_gap"].transform(
        lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0.0
    )
    return df


def flag_sentiment_drops(
    quarterly_df: pd.DataFrame,
    threshold_zscore: float = 1.5,
    ticker_col: str = "ticker",
    score_col: str = "avg_sentiment",
) -> pd.DataFrame:
    """
    Flag quarters where combined sentiment dropped more than `threshold_zscore`
    standard deviations below that ticker's historical mean.

    Z-scores are computed within-ticker, not cross-sectionally, a z = −1.5 means
    the quarter is 1.5 SD below that company's own historical average.
    """
    df = quarterly_df.copy()
    df["z_score"] = df.groupby(ticker_col)[score_col].transform(
        lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0.0
    )
    df["anomaly"] = df["z_score"] < -threshold_zscore
    df["qoq_change"] = df.groupby(ticker_col)[score_col].diff()
    return df


def pivot_heatmap(
    quarterly_df: pd.DataFrame,
    ticker_col: str = "ticker",
    score_col: str = "avg_sentiment",
) -> pd.DataFrame:
    """Pivot to (ticker × quarter) matrix for heatmap plotting."""
    return quarterly_df.pivot_table(
        index=ticker_col,
        columns="year_quarter",
        values=score_col,
    )


def disaggregate_by_tier(
    sentiment_df: pd.DataFrame,
    tier_col: str = "tier",
    date_col: str = "filing_date",
    score_col: str = "score",
) -> pd.DataFrame:
    """
    Aggregate quarterly sentiment by tier (GSIB / super_regional / stressed).

    Returns a DataFrame with columns: tier, year_quarter, avg_sentiment, n_tickers
    """
    df = sentiment_df.copy()
    if tier_col not in df.columns:
        return pd.DataFrame()
    df[date_col] = pd.to_datetime(df[date_col])
    df["year_quarter"] = df[date_col].dt.to_period("Q")

    agg = (
        df.groupby([tier_col, "year_quarter"])
        .agg(avg_sentiment=(score_col, "mean"), n_docs=(score_col, "count"))
        .reset_index()
    )
    return agg.sort_values([tier_col, "year_quarter"])


def survivorship_analysis(
    sentiment_df: pd.DataFrame,
    failed_tickers: list[str],
    ticker_col: str = "ticker",
    date_col: str = "filing_date",
    score_col: str = "score",
) -> pd.DataFrame:
    """
    Compare sector-average sentiment computed with vs without failed banks.

    Returns a DataFrame with columns:
      year_quarter, avg_sentiment_all, avg_sentiment_survivors, survivorship_bias
    where survivorship_bias = survivors_average − all_average (positive = inflated)
    """
    df = sentiment_df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df["year_quarter"] = df[date_col].dt.to_period("Q")

    all_avg = (
        df.groupby("year_quarter")[score_col].mean()
        .reset_index()
        .rename(columns={score_col: "avg_sentiment_all"})
    )
    surv_avg = (
        df[~df[ticker_col].isin(failed_tickers)]
        .groupby("year_quarter")[score_col].mean()
        .reset_index()
        .rename(columns={score_col: "avg_sentiment_survivors"})
    )
    merged = all_avg.merge(surv_avg, on="year_quarter", how="left")
    merged["survivorship_bias"] = merged["avg_sentiment_survivors"] - merged["avg_sentiment_all"]
    return merged.sort_values("year_quarter")
