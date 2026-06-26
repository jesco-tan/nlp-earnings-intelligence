"""Smoke tests for nlp-earnings-intelligence, Tier 1 only (no NLP dependencies)."""

import pandas as pd
import pytest

from src.nlp_ei.synthetic import generate_synthetic_corpus
from src.nlp_ei.sentiment import score_mock, run_sentiment
from src.nlp_ei.timeseries import (
    aggregate_quarterly,
    aggregate_quarterly_by_section,
    compute_section_gap,
    flag_sentiment_drops,
    pivot_heatmap,
    disaggregate_by_tier,
    survivorship_analysis,
)
from src.nlp_ei.ner import _mock_entities


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def corpus_uniform():
    """Small uniform corpus with known generic tickers (backward-compatible mode)."""
    return generate_synthetic_corpus(
        tickers=["AAPL", "MSFT", "JPM"],
        quarters_per_ticker=4,
        seed=0,
    )


@pytest.fixture
def corpus_scenario():
    """Full scenario corpus with banking tickers."""
    return generate_synthetic_corpus(
        tickers=["JPM", "BAC", "SIVB", "FRC"],
        start_date="2020-01-01",
        end_date="2023-06-30",
        seed=42,
    )


# ---------------------------------------------------------------------------
# Corpus generation
# ---------------------------------------------------------------------------

def test_corpus_schema_uniform(corpus_uniform):
    assert len(corpus_uniform) > 0
    required = {"ticker", "filing_date", "text", "section"}
    assert required.issubset(corpus_uniform.columns)
    assert set(corpus_uniform["ticker"]) == {"AAPL", "MSFT", "JPM"}


def test_corpus_schema_scenario(corpus_scenario):
    required = {"ticker", "filing_date", "text", "section", "true_sentiment",
                "tier", "balance_sheet_health", "rate_regime", "failed_bank"}
    assert required.issubset(corpus_scenario.columns)
    assert corpus_scenario["true_sentiment"].between(-1.0, 1.0).all()


def test_failed_banks_stop_before_failure(corpus_scenario):
    """SIVB and FRC should not have filings in or after 2023Q1."""
    for ticker in ["SIVB", "FRC"]:
        sub = corpus_scenario[corpus_scenario["ticker"] == ticker]
        if len(sub) > 0:
            max_period = sub["year_quarter"].max()
            assert max_period <= pd.Period("2022Q4"), (
                f"{ticker} should stop filing before 2023Q1, got {max_period}"
            )


def test_scenario_has_two_sections(corpus_scenario):
    assert set(corpus_scenario["section"]) == {"mda", "risk_factors"}


def test_tier_structure(corpus_scenario):
    gsib = corpus_scenario[corpus_scenario["ticker"].isin(["JPM", "BAC"])]["tier"].unique()
    assert all(t == "gsib" for t in gsib)


# ---------------------------------------------------------------------------
# Mock sentiment scorer
# ---------------------------------------------------------------------------

def test_mock_sentiment_no_true_sentiments(corpus_uniform):
    texts = corpus_uniform["text"].tolist()
    scores = score_mock(texts)
    assert len(scores) == len(texts)
    assert all(0 <= s["positive"] <= 1 for s in scores)
    assert all(-1 <= s["score"] <= 1 for s in scores)
    assert all(s["label"] in ("positive", "negative") for s in scores)


def test_mock_sentiment_with_true_sentiments(corpus_scenario):
    texts = corpus_scenario["text"].tolist()
    true_sents = corpus_scenario["true_sentiment"].tolist()
    scores = score_mock(texts, true_sentiments=true_sents)
    assert len(scores) == len(texts)
    # Check correlation direction: positive true_sentiment → positive score
    import numpy as np
    scored = [s["score"] for s in scores]
    corr = float(np.corrcoef(true_sents, scored)[0, 1])
    assert corr > 0.5, f"score_mock correlation with true_sentiment too low: {corr:.3f}"


# ---------------------------------------------------------------------------
# End-to-end sentiment pipeline
# ---------------------------------------------------------------------------

def test_run_sentiment_uniform(corpus_uniform):
    result = run_sentiment(corpus_uniform, use_finbert=False)
    assert "score" in result.columns
    assert len(result) == len(corpus_uniform)


def test_run_sentiment_preserves_true_sentiment(corpus_scenario):
    result = run_sentiment(corpus_scenario, use_finbert=False)
    assert "score" in result.columns
    import numpy as np
    corr = float(np.corrcoef(result["true_sentiment"], result["score"])[0, 1])
    assert corr > 0.4, f"run_sentiment correlation too low: {corr:.3f}"


# ---------------------------------------------------------------------------
# Quarterly aggregation
# ---------------------------------------------------------------------------

def test_aggregate_quarterly(corpus_uniform):
    with_scores = run_sentiment(corpus_uniform, use_finbert=False)
    quarterly = aggregate_quarterly(with_scores)
    assert "avg_sentiment" in quarterly.columns
    assert quarterly["ticker"].nunique() == 3


def test_aggregate_quarterly_by_section(corpus_scenario):
    with_scores = run_sentiment(corpus_scenario, use_finbert=False)
    sec_q = aggregate_quarterly_by_section(with_scores)
    assert "mda_sentiment" in sec_q.columns
    assert "risk_sentiment" in sec_q.columns


def test_compute_section_gap(corpus_scenario):
    with_scores = run_sentiment(corpus_scenario, use_finbert=False)
    sec_q = aggregate_quarterly_by_section(with_scores)
    gap_df = compute_section_gap(sec_q)
    assert "section_gap" in gap_df.columns
    # For MD&A > risk_factors structurally, gap should be positive on average
    assert gap_df["section_gap"].mean() > 0


# ---------------------------------------------------------------------------
# Anomaly detection
# ---------------------------------------------------------------------------

def test_anomaly_flagging(corpus_uniform):
    with_scores = run_sentiment(corpus_uniform, use_finbert=False)
    quarterly = aggregate_quarterly(with_scores)
    flagged = flag_sentiment_drops(quarterly, threshold_zscore=0.5)
    assert "anomaly" in flagged.columns
    assert "z_score" in flagged.columns


# ---------------------------------------------------------------------------
# Disaggregation
# ---------------------------------------------------------------------------

def test_disaggregate_by_tier(corpus_scenario):
    with_scores = run_sentiment(corpus_scenario, use_finbert=False)
    tier_agg = disaggregate_by_tier(with_scores)
    assert "tier" in tier_agg.columns
    assert "avg_sentiment" in tier_agg.columns
    assert tier_agg["tier"].nunique() >= 2


def test_survivorship_analysis(corpus_scenario):
    with_scores = run_sentiment(corpus_scenario, use_finbert=False)
    surv = survivorship_analysis(with_scores, failed_tickers=["SIVB", "FRC"])
    assert "avg_sentiment_all" in surv.columns
    assert "avg_sentiment_survivors" in surv.columns
    assert "survivorship_bias" in surv.columns


# ---------------------------------------------------------------------------
# Pivot heatmap
# ---------------------------------------------------------------------------

def test_pivot_heatmap(corpus_uniform):
    with_scores = run_sentiment(corpus_uniform, use_finbert=False)
    quarterly = aggregate_quarterly(with_scores)
    hm = pivot_heatmap(quarterly)
    assert hm.shape[0] == 3


# ---------------------------------------------------------------------------
# NER mock
# ---------------------------------------------------------------------------

def test_mock_entities_schema(corpus_scenario):
    entities = _mock_entities(corpus_scenario)
    assert {"entity", "label", "count"}.issubset(entities.columns)
    assert len(entities) > 0
    assert (entities["count"] > 0).all()
