"""
Full analysis pipeline for the regional bank stress scenario.
Generates all outputs/ artifacts and run_summary.json.
Run from project root: .venv/Scripts/python.exe scripts/run_pipeline.py
"""

import sys
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import yaml
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.nlp_ei.synthetic import generate_synthetic_corpus
from src.nlp_ei.sentiment import run_sentiment
from src.nlp_ei.ner import extract_entities, top_entities_by_type
from src.nlp_ei.topics import mock_topics
from src.nlp_ei.timeseries import (
    aggregate_quarterly,
    aggregate_quarterly_by_section,
    compute_section_gap,
    flag_sentiment_drops,
    pivot_heatmap,
    disaggregate_by_tier,
    survivorship_analysis,
)
from src.nlp_ei.visualise import (
    plot_sentiment_heatmap,
    plot_sentiment_trends,
    plot_entity_freq,
    plot_section_gap,
    plot_tier_comparison,
    plot_score_distribution,
    plot_survivorship,
)

OUTPUTS = ROOT / "outputs"
OUTPUTS.mkdir(exist_ok=True)

cfg = yaml.safe_load((ROOT / "config" / "default.yaml").read_text())
edgar_cfg = cfg["edgar"]
nlp_cfg = cfg["nlp"]
anomaly_cfg = cfg["anomaly"]
scenario_cfg = cfg["scenario"]

FAILED_BANKS = scenario_cfg["failed_banks"]
GAP_ALERT = anomaly_cfg["section_gap_alert_threshold"]
ZSCORE_THRESHOLD = anomaly_cfg["sentiment_zscore_threshold"]

print("=" * 60)
print("NLP Earnings Intelligence, Regional Bank Stress Pipeline")
print("=" * 60)

# ── 1. Corpus ─────────────────────────────────────────────────
print("\n[1/8] Generating synthetic corpus ...")
corpus = generate_synthetic_corpus(
    tickers=edgar_cfg["tickers"],
    start_date=edgar_cfg["date_range"]["start"],
    end_date=edgar_cfg["date_range"]["end"],
    seed=42,
)
print(f"  {len(corpus):,} documents | {corpus['ticker'].nunique()} tickers | "
      f"{corpus['year_quarter'].nunique()} distinct quarters")
print(f"  Sections: {sorted(corpus['section'].unique())}")
print(f"  Tiers: {corpus.groupby('tier')['ticker'].apply(lambda x: sorted(x.unique())).to_dict()}")
failed_doc_count = corpus[corpus["failed_bank"]]["ticker"].nunique()
print(f"  Failed-bank tickers: {FAILED_BANKS} "
      f"({corpus[corpus['failed_bank']]['year_quarter'].max()} last quarter)")

# ── 2. Sentiment ───────────────────────────────────────────────
print("\n[2/8] Scoring sentiment (mock, correlated with true_sentiment) ...")
USE_FINBERT = False  # Set True if transformers + torch installed
sentiment_df = run_sentiment(corpus, use_finbert=USE_FINBERT,
                             model_name=nlp_cfg["finbert_model"],
                             batch_size=nlp_cfg["sentence_batch_size"])
corr = float(np.corrcoef(sentiment_df["true_sentiment"], sentiment_df["score"])[0, 1])
print(f"  Mock scorer <-> true_sentiment correlation: r = {corr:.3f}")
print(f"  Mean score, MD&A: {sentiment_df[sentiment_df['section']=='mda']['score'].mean():.3f}  "
      f"Risk: {sentiment_df[sentiment_df['section']=='risk_factors']['score'].mean():.3f}")

# ── 3. Section-level aggregation & gap ────────────────────────
print("\n[3/8] Aggregating by section and computing MD&A–risk gap ...")
section_quarterly = aggregate_quarterly_by_section(sentiment_df)
section_quarterly = compute_section_gap(section_quarterly)

# Add tier column
tier_map = corpus.drop_duplicates("ticker").set_index("ticker")["tier"].to_dict()
section_quarterly["tier"] = section_quarterly["ticker"].map(tier_map)

# Find max gap per ticker
gap_by_ticker = (
    section_quarterly.groupby("ticker")["section_gap"]
    .agg(["max", "mean"])
    .rename(columns={"max": "max_gap", "mean": "mean_gap"})
    .round(3)
)
print(gap_by_ticker.to_string())
max_gap_row = section_quarterly.loc[section_quarterly["section_gap"].idxmax()]
print(f"\n  Widest gap: {max_gap_row['ticker']} in {max_gap_row['year_quarter']} "
      f"= {max_gap_row['section_gap']:.3f}")

alert_rows = section_quarterly[section_quarterly["section_gap"] > GAP_ALERT]
print(f"  Quarters exceeding gap-alert ({GAP_ALERT}): {len(alert_rows)} "
      f"({alert_rows['ticker'].value_counts().to_dict()})")

# ── 4. Quarterly aggregation & anomalies ──────────────────────
print("\n[4/8] Quarterly aggregation and anomaly detection ...")
quarterly = aggregate_quarterly(sentiment_df)
# Attach tier
quarterly["tier"] = quarterly["ticker"].map(tier_map)
anomaly_df = flag_sentiment_drops(quarterly, threshold_zscore=ZSCORE_THRESHOLD)
anomaly_df["tier"] = anomaly_df["ticker"].map(tier_map)

flagged = anomaly_df[anomaly_df["anomaly"]]
n_anomalies = int(len(flagged))
print(f"  Anomaly quarters detected (z < −{ZSCORE_THRESHOLD}): {n_anomalies}")
if n_anomalies:
    print(flagged[["ticker", "year_quarter", "avg_sentiment", "z_score"]].to_string(index=False))

# Tier-level sentiment summary
tier_summary = (
    anomaly_df.groupby("tier")["avg_sentiment"]
    .agg(["mean", "std", "min"])
    .round(3)
    .rename(columns={"mean": "mean_sentiment", "std": "sd", "min": "min_sentiment"})
)
print("\n  Tier-level sentiment:\n", tier_summary.to_string())

# Section-level means by tier
for tier in ["gsib", "super_regional", "stressed"]:
    tickers_in_tier = [t for t, v in tier_map.items() if v == tier]
    sub = sentiment_df[sentiment_df["ticker"].isin(tickers_in_tier)]
    m = sub.groupby("section")["score"].mean()
    mda_m = m.get("mda", float("nan"))
    risk_m = m.get("risk_factors", float("nan"))
    print(f"  {tier:20s}  MD&A={mda_m:.3f}  risk={risk_m:.3f}  gap={mda_m-risk_m:.3f}")

# ── 5. Survivorship analysis ──────────────────────────────────
print("\n[5/8] Survivorship bias analysis ...")
surv_df = survivorship_analysis(sentiment_df, failed_tickers=FAILED_BANKS)
# Post-failure quarters only
post_failure = surv_df[surv_df["year_quarter"] >= pd.Period("2023Q1")]
if not post_failure.empty:
    mean_bias = post_failure["survivorship_bias"].mean()
    print(f"  Post-failure survivorship bias (mean): +{mean_bias:.3f} "
          f"(sector average inflated by this amount without failed banks)")
else:
    mean_bias = float(
        surv_df["survivorship_bias"].dropna().mean()
    )
    print(f"  Mean survivorship bias across all quarters: +{mean_bias:.3f}")

# Overall sector with/without
sector_with = float(surv_df["avg_sentiment_all"].mean())
sector_without = float(surv_df["avg_sentiment_survivors"].mean())
print(f"  Sector mean (all): {sector_with:.3f}  |  (survivors): {sector_without:.3f}")

# ── 6. NER ────────────────────────────────────────────────────
print("\n[6/8] Named entity recognition ...")
entity_df = extract_entities(corpus, spacy_model=nlp_cfg["spacy_model"])
print(f"  Entity types: {sorted(entity_df['label'].unique())}")
print(f"  Top entities:\n{entity_df.head(8).to_string(index=False)}")

# ── 7. Topic modelling ────────────────────────────────────────
print("\n[7/8] Topic modelling (mock) ...")
USE_BERTOPIC = False
texts = corpus["text"].tolist()
topic_df = mock_topics(texts)
print(topic_df["topic_label"].value_counts().to_string())

# ── 8. Visualisations ─────────────────────────────────────────
print("\n[8/8] Generating visualisations ...")
plt.rcParams.update({"figure.dpi": 150})

hm = pivot_heatmap(quarterly)
plot_sentiment_heatmap(hm, save_path=OUTPUTS / "sentiment_heatmap.png")
print("  sentiment_heatmap.png")

plot_sentiment_trends(quarterly, anomaly_df=anomaly_df,
                      save_path=OUTPUTS / "sentiment_trends.png")
print("  sentiment_trends.png")

entity_df.to_csv(OUTPUTS / "entity_freq.csv", index=False)
plot_entity_freq(entity_df, top_n=15, save_path=OUTPUTS / "entity_freq.png")
print("  entity_freq.csv / entity_freq.png")

plot_section_gap(section_quarterly, gap_alert=GAP_ALERT,
                 save_path=OUTPUTS / "section_gap.png")
print("  section_gap.png")

tier_agg = disaggregate_by_tier(sentiment_df)
plot_tier_comparison(tier_agg, save_path=OUTPUTS / "tier_comparison.png")
print("  tier_comparison.png")

plot_score_distribution(sentiment_df, save_path=OUTPUTS / "score_distribution.png")
print("  score_distribution.png")

plot_survivorship(surv_df, save_path=OUTPUTS / "survivorship.png")
print("  survivorship.png")

# ── Export ────────────────────────────────────────────────────
quarterly.to_csv(OUTPUTS / "quarterly_sentiment.csv", index=False)
sentiment_df.to_csv(OUTPUTS / "sentiment_scored.csv", index=False)

# Compute key stats for run_summary
gsib_tickers = [t for t, v in tier_map.items() if v == "gsib"]
sr_tickers = [t for t, v in tier_map.items() if v == "super_regional"]
stressed_tickers = [t for t, v in tier_map.items() if v == "stressed"]

def _tier_mean(tickers, section=None):
    sub = sentiment_df[sentiment_df["ticker"].isin(tickers)]
    if section:
        sub = sub[sub["section"] == section]
    return round(float(sub["score"].mean()), 4)

sivb_anomalies = flagged[flagged["ticker"] == "SIVB"]
frc_anomalies = flagged[flagged["ticker"] == "FRC"]
sivb_last_anom = str(sivb_anomalies["year_quarter"].max()) if len(sivb_anomalies) else "none"
frc_last_anom = str(frc_anomalies["year_quarter"].max()) if len(frc_anomalies) else "none"

run_summary = {
    "schema_version": 2,
    "entity_method": "recorded independently by entity stage; not FinBERT",
    "topic_method": "random mock labels, seed 0",
    "scenario": scenario_cfg["name"],
    "n_documents": int(len(corpus)),
    "n_tickers": int(corpus["ticker"].nunique()),
    "n_quarters_analyzed": int(corpus["year_quarter"].nunique()),
    "data_source": "synthetic",
    "sentiment_method": "FinBERT" if USE_FINBERT else "mock (correlated with true_sentiment)",
    "mock_vs_true_correlation": round(corr, 4),
    "tiers": {
        "gsib": gsib_tickers,
        "super_regional": sr_tickers,
        "stressed": stressed_tickers,
    },
    "failed_banks": FAILED_BANKS,
    "failed_banks_last_quarter": "2022Q4",
    "n_anomaly_quarters": n_anomalies,
    "anomaly_tickers": sorted(flagged["ticker"].unique().tolist()) if n_anomalies else [],
    "sivb_last_anomaly_quarter": sivb_last_anom,
    "frc_last_anomaly_quarter": frc_last_anom,
    "mean_mda_sentiment_gsib": _tier_mean(gsib_tickers, "mda"),
    "mean_risk_sentiment_gsib": _tier_mean(gsib_tickers, "risk_factors"),
    "mean_mda_sentiment_stressed": _tier_mean(stressed_tickers, "mda"),
    "mean_risk_sentiment_stressed": _tier_mean(stressed_tickers, "risk_factors"),
    "max_section_gap_ticker": str(max_gap_row["ticker"]),
    "max_section_gap_quarter": str(max_gap_row["year_quarter"]),
    "max_section_gap_value": round(float(max_gap_row["section_gap"]), 4),
    "section_gap_alert_threshold": GAP_ALERT,
    "gap_alert_ticker_counts": alert_rows["ticker"].value_counts().to_dict(),
    "sector_mean_sentiment_all_banks": round(sector_with, 4),
    "sector_mean_sentiment_survivors_only": round(sector_without, 4),
    "historical_cohort_difference_score": round(sector_without - sector_with, 4),
    "zscore_threshold": ZSCORE_THRESHOLD,
}

(OUTPUTS / "run_summary.json").write_text(json.dumps(run_summary, indent=2, default=str))
print("\n  run_summary.json")
print("\n" + "=" * 60)
print("Pipeline complete. Outputs in outputs/")
print("=" * 60)
print(json.dumps(run_summary, indent=2, default=str))
