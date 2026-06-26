# NLP Earnings Intelligence: Regional Bank Stress (2020–2024)

**Headline result**: FinBERT-derived z-score anomaly detection flagged Silicon Valley Bank (z = −1.91) and First Republic Bank (z = −1.71) in Q4 2022, one quarter before their March and May 2023 failures. SIVB's section-divergence signal (MD&A minus risk-factors sentiment) had already breached the 0.60 alert threshold in three consecutive quarters through 2022 Q3 (peak: 0.925 in 2021 Q2), providing earlier structural warning than the z-score alone.

End-to-end NLP pipeline on SEC 10-K/10-Q filings covering eight U.S. banks across the COVID shock, 525 bp rate hiking cycle, and 2023 banking crisis. Extracts FinBERT sentiment, BERTopic latent topics, and spaCy named entities; aggregates to quarterly time series; flags anomaly drops; and audits for survivorship bias in sector-level averages.

---

## Techniques demonstrated

| Technique | Implementation | Notes |
|---|---|---|
| **FinBERT sentiment** | `src/nlp_ei/sentiment.py` | 512-token chunking, chunk-averaged; mock scorer correlates at r = 0.91 with planted ground truth |
| **Named entity recognition** | `src/nlp_ei/ner.py` | spaCy `en_core_web_sm`; banking mock with ORG/GPE/PRODUCT/MONEY/PERCENT |
| **BERTopic topic modelling** | `src/nlp_ei/topics.py` | `all-MiniLM-L6-v2` embeddings + HDBSCAN; 7 banking-domain mock topics in Tier 1 |
| **Section-level divergence** | `src/nlp_ei/timeseries.py:compute_section_gap` | MD&A minus risk-factors per quarter; z-scores within ticker |
| **Anomaly detection** | `src/nlp_ei/timeseries.py:flag_sentiment_drops` | Within-ticker z-score; threshold −1.5 SD |
| **Tier disaggregation** | `src/nlp_ei/timeseries.py:disaggregate_by_tier` | GSIB / super-regional / stressed breakdowns |
| **Survivorship-bias audit** | `src/nlp_ei/timeseries.py:survivorship_analysis` | Sector average with vs without failed banks |
| **Tiered dependency design** | `src/nlp_ei/*.py` | Full pipeline at three dependency levels; mocks mirror real function signatures exactly |

---

## Notebook sections

| Section | What it answers |
|---|---|
| 0. Setup | Config and imports |
| 1. Banking landscape | Why these banks, what the rate cycle looked like |
| 2. Corpus assembly | Document counts, ticker coverage, failed-bank cutoffs |
| 3. Sentiment scoring | FinBERT / mock results; section-level means |
| 4. Score distributions | MD&A vs risk-factors distributional difference; tier separation |
| 5. MD&A–risk divergence | Section gap by ticker; alert threshold flags; the NIM/risk tension |
| 6. Quarterly anomaly detection | z-score time series; SIVB and FRC pre-failure flags |
| 7. Tier disaggregation | GSIB / super-regional / stressed quarterly comparison |
| 8. NER | Banking entity frequency; geographic and institutional mentions |
| 9. Topic modelling | BERTopic cluster map (Tier 3) or mock topics (Tier 1) |
| 10. Calibration | Threshold sensitivity; signal-to-noise at different z thresholds |
| 11. Survivorship-bias analysis | Sector average with vs without failed banks (+2.6 pp bias) |
| 12. Recommendation & synthesis | Per-tier decision table; analytical tension resolved |
| 13. Export | CSV / JSON outputs |

---

## Output files

| File | Description |
|---|---|
| `outputs/sentiment_heatmap.png` | Ticker × quarter net sentiment grid |
| `outputs/sentiment_trends.png` | Per-ticker trends with anomaly markers (red dashes) |
| `outputs/section_gap.png` | MD&A vs risk-factors divergence, the early warning chart |
| `outputs/tier_comparison.png` | GSIB / super-regional / stressed quarterly series |
| `outputs/score_distribution.png` | Sentiment distributions by section type and tier |
| `outputs/survivorship.png` | Sector average with vs without failed banks |
| `outputs/entity_freq.png` / `.csv` | Top 15 banking entities by mention count |
| `outputs/quarterly_sentiment.csv` | Quarterly aggregated scores with z-scores |
| `outputs/sentiment_scored.csv` | Per-document FinBERT scores |
| `outputs/run_summary.json` | Key numeric results |
| `outputs/report.html` | Self-contained stakeholder report (images base64-embedded) |

---

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux
pip install -r requirements.txt

# Run the full pipeline (generates all outputs/)
python scripts/run_pipeline.py

# Open the notebook
jupyter notebook notebooks/nlp_walkthrough.ipynb

# Generate the HTML stakeholder report
python scripts/generate_html_report.py
```

Tests (Tier 1 only, no NLP dependencies):
```bash
pytest tests -q
```

---

## Enabling real NLP (Tier 2)

```bash
pip install transformers torch spacy
python -m spacy download en_core_web_sm
# In the notebook: USE_FINBERT = True
```

## Enabling BERTopic + real EDGAR data (Tier 3)

```bash
pip install bertopic sentence-transformers umap-learn hdbscan sec-edgar-downloader beautifulsoup4
# In the notebook: USE_BERTOPIC = True, USE_SYNTHETIC = False
```

FinBERT downloads ~440 MB on first use (cached to `~/.cache/huggingface/`).

---

## Scenario

**Eight U.S. regional banks, 2020–2024.** Three balance-sheet-health tiers:

- **GSIB** (JPM, BAC): diversified, high capital, resilient sentiment throughout
- **Super-Regional** (USB, RF): moderate stress in 2022–2023, recovered in 2024
- **Stressed** (KEY, ZION, SIVB, FRC): concentrated deposits, long-duration bonds
  SIVB and FRC fail in Q1 2023

Synthetic corpus with latent `balance_sheet_health` variable that drives diverging
MD&A and risk-factor trajectories through the rate cycle. The NIM boost in 2022
temporarily inflates MD&A positivity for low-health banks while risk language tightens ,
creating the "optimism gap" that the section-divergence signal is designed to detect.

See [`REPORT.md`](REPORT.md) for the full analyst memo with all statistics.

---

## Tech stack

`transformers` (FinBERT) · `spaCy` · `BERTopic` · `sentence-transformers` · `sec-edgar-downloader` · `pandas` · `matplotlib` · `seaborn` · `plotly`
