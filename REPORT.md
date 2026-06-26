# Analyst Memo: NLP Earnings Intelligence, Regional Bank Stress 2020–2024

**Date**: 2026-06-24
**Analyst**: Jesco Tan
**Coverage**: JPM, BAC, USB, RF, KEY, ZION, SIVB, FRC
**Period**: 2020 Q1 – 2024 Q1

---

## Executive Summary

The z-score anomaly detection system flagged Silicon Valley Bank (z = −1.91) and First Republic Bank (z = −1.71) in Q4 2022, one quarter before their March and May 2023 failures. A complementary section-divergence signal, the gap between MD&A sentiment (+0.16 average for stressed banks) and risk-factors sentiment (−0.26), peaked at 0.925 for SIVB in 2021 Q2 and remained above the 0.60 alert threshold through 2022 Q3, more than double the GSIB average gap of 0.328. These two signals together provide convergent evidence that SIVB and FRC were structural outliers: the gap fired first (as early as 2021 Q2), and the z-score confirmed the deterioration in 2022 Q4.

The central complication: sector-average sentiment shows apparent improvement in 2023–2024. This is survivorship bias. Excluding the two failed banks inflates the sector average by +2.6 percentage points (sector mean with all banks: 0.015; survivors only: 0.042). A naive analyst reading the aggregate trend as "sector recovery" is looking at a sample that mechanically improved because the worst performers were removed. The per-ticker anomaly signals are correct; the sector average is misleading.

**Recommendation**: Use per-ticker z-scores and section-gap signals for early warning. Apply survivorship-bias adjustment to any sector-level benchmarking. The 2024 GSIB recovery (mean sentiment +0.114) is real; the stressed-tier "recovery" partly reflects survivorship and should be treated as such.

---

## 1. Scenario and Data

**Eight banks, three tiers, five rate regimes.** The Federal Reserve cut rates to zero in March 2020, held them through 2021, then raised them 525 basis points from March 2022 to July 2023, the fastest tightening in forty years. Eight banks were selected to span the balance-sheet-health spectrum:

| Tier | Tickers | Balance-Sheet Health | Outcome |
|---|---|---|---|
| GSIB | JPM, BAC | 0.82–0.88 | Survived |
| Super-Regional | USB, RF | 0.63–0.68 | Survived |
| Stressed | KEY, ZION | 0.48–0.52 | Survived |
| Stressed (failed) | SIVB, FRC | 0.25–0.31 | Failed 2023 |

The corpus contains **252 filing sections** (8 tickers × varying quarters × 2 sections, MD&A and Risk Factors) covering **17 calendar quarters**. SIVB and FRC filed through 2022 Q4 only.

The synthetic corpus plants a latent `balance_sheet_health` variable per ticker that drives sentiment through a parametric model: health → MD&A trajectory and health → risk-factors trajectory, with the hiking regime adding a NIM boost to MD&A for low-health banks that does not appear in risk language. This creates the structural divergence the pipeline is designed to detect.

---

## 2. Sentiment Scoring

**FinBERT** (`ProsusAI/finbert`) scores each section as positive/negative/neutral; net sentiment = positive − negative, range [−1, +1]. Long sections are chunked at ≤512 tokens and chunk scores are averaged.

At Tier 1 (no ML dependencies), a mock scorer calibrated to the planted `true_sentiment` produces scores with **r = 0.91 correlation** with ground truth. This preserves all structural patterns, section-level, tier-level, and temporal, through the pipeline.

**Section-level means** across all tickers and quarters:
- MD&A: mean = **+0.202**, std = 0.154
- Risk Factors: mean = **−0.177**, std = 0.171

MD&A is structurally more positive because management language in the operating review is optimistic by convention; risk disclosures use hedged, forward-looking language. This structural difference is the baseline from which the section gap is measured.

**Tier-level means** (combined across sections):
- GSIB: **+0.114** (std = 0.136)
- Super-Regional: **+0.021** (std = 0.139)
- Stressed: **−0.052** (std = 0.139)

The 0.166-point spread between GSIB and stressed tier is economically meaningful, even on 252 documents.

---

## 3. Section-Level Divergence (The Guardrail)

The *section gap* = MD&A sentiment − risk-factors sentiment. A large positive gap means management is telling an optimistic operating story while simultaneously escalating risk disclosures, a sign of optimism bias or management not updating the MD&A narrative as fast as the disclosure lawyers are updating the risk section.

**Alert threshold**: gap > 0.60 flags a quarter for review. Below 0.60, the gap is consistent with normal section-type structural difference. Above it, risk language has deteriorated faster than the MD&A narrative reflects.

**Tier-level section gaps**:
- GSIB: MD&A +0.278, risk −0.049, **gap = 0.328** (narrow, stable)
- Super-Regional: MD&A +0.203, risk −0.160, **gap = 0.363** (moderate)
- Stressed: MD&A +0.158, risk −0.262, **gap = 0.419** (wide)

The stressed tier's wider mean gap reflects the latent structure: low-health banks have steeper risk-language deterioration relative to their MD&A. In the hiking regime, the NIM boost to MD&A temporarily widens this further.

**Alert-threshold breaches** (gap > 0.60 by ticker):
- ZION: 4 quarters
- KEY: 3 quarters
- SIVB: 3 quarters (2021 Q2, 2022 Q2, 2022 Q3)
- RF: 2 quarters
- FRC: 2 quarters (2022 Q2, 2022 Q4)
- JPM: 2 quarters; USB: 2 quarters; BAC: 1 quarter
- Maximum observed: KEY in 2022 Q4 = **0.972**; SIVB in 2021 Q2 = **0.925**

An important nuance: SIVB's peak section gap (0.925) occurred in 2021 Q2 and remained elevated through 2022 Q3 (0.716), but had already compressed to 0.489 in 2022 Q4 as MD&A sentiment also turned negative. The gap signal provided earlier warning (firing from 2021 Q2 onward) than the z-score (which fired in 2022 Q4). In 2022 Q4, the z-score anomaly fired because both sections deteriorated simultaneously, collapsing the gap but dragging the combined average to its lowest level.

The mechanistic reason SIVB shows the largest gaps through 2022 Q3 is the same NIM expansion dynamic: rising rates kept MD&A language upbeat while risk language escalated around deposit concentration and unrealized HTM securities losses. The gap compresses in 2022 Q4 not because the risks resolved, but because the MD&A narrative had by then also turned negative, which is itself an escalation signal.

---

## 4. Quarterly Anomaly Detection

Within-ticker z-scores flag quarters where combined sentiment falls > 1.5 standard deviations below that ticker's own historical mean. **11 quarters flagged** across 7 tickers.

**Pre-failure signals (the primary result)**:
- SIVB, 2022 Q4: z = **−1.905**, avg sentiment = −0.394 (lowest in the sample)
- FRC, 2022 Q4: z = **−1.711**, avg sentiment = −0.318

Both SIVB and FRC were flagged in Q4 2022, **one quarter before their March and May 2023 failures**. The z-score for SIVB at −1.91 is the most extreme in the entire sample.

**COVID-era flag**:
- FRC, 2020 Q3: z = −1.540, avg sentiment = −0.296

**COVID-era flags (expected, not false positives)**:
Seven banks were flagged in the 2020 Q2–Q4 window. This is mechanistically correct, the COVID stress event involved massive loan-loss provisioning, NIM compression from emergency rate cuts, and genuine uncertainty about borrower performance. These were real anomalies. An analyst would classify COVID-era flags separately from the 2022 structural flags.

**An important nuance about the anomaly detector**: It catches SIVB in Q4 2022 and FRC in Q3–Q4 2022, but also flags some super-regional and GSIB quarters during COVID. This is expected behavior: the detector is designed to flag genuine anomalies, not only failures. The analyst's job is to investigate the cause after the flag is raised.

---

## 5. Tier Disaggregation

**Why GSIBs are different**: JPM and BAC maintained positive combined sentiment (+0.114 mean) throughout the cycle. Their MD&A language reflects diversified revenue streams and strong capital ratios; their risk language, while escalating in 2022, did not reach the severity seen in stressed-tier filings. The mechanistic reason is capital buffer: GSIBs can absorb unrealized HTM losses without facing deposit-flight pressure because their capital ratios provide confidence to depositors and counterparties.

**Why the stressed tier's MD&A stayed elevated in 2022**: Net interest margin genuinely expanded as short-term rates rose faster than deposit costs. SIVB and FRC were reporting real NIM improvement in 2022, their MD&A optimism was not fabricated. The problem was that their deposit beta assumptions proved wrong (deposits fled faster than historical models suggested when SVB disclosed its balance-sheet repositioning in March 2023) and their HTM portfolio crystallized losses that were previously invisible under hold-to-maturity accounting.

**Survivor bias within the stressed tier**: KEY and ZION show section gaps up to 0.810–0.972 but survived. This illustrates that a wide section gap is a *risk indicator*, not a deterministic failure predictor. KEY and ZION had different deposit bases and were not as concentrated in the technology-sector venture-capital ecosystem. The section gap identifies banks worth deeper investigation, not a binary pass/fail signal.

---

## 6. Named Entity Recognition

Top banking-domain entities by mention count across 252 documents:
- **Federal Reserve** (596 mentions): rate-risk language dominates the corpus
- **United States** (655): geographic baseline in risk disclosures
- **commercial real estate** (426): post-hike refinancing risk, most prominent in 2022–2024
- **held-to-maturity** (373): unrealized-loss language, specific to the hiking regime
- **FDIC** (327): deposit insurance mentions, rising in 2022–2023 stress
- **California** (392): geographic deposit concentration signal for SIVB

The prominence of "held-to-maturity" as a product entity is specific to the hiking-regime period and would not appear in a pre-2022 corpus at the same frequency. This temporal shift in entity prominence is itself an NLP signal: when a previously obscure accounting category becomes one of the most mentioned products in risk disclosures, it warrants attention.

---

## 7. Topic Modelling

BERTopic (Tier 3) clusters filing sections into latent themes using HDBSCAN over `all-MiniLM-L6-v2` embeddings. The mock distribution on the Tier 1 synthetic corpus assigns:
1. Deposit stability / liquidity stress (51 documents, 20%)
2. Net interest margin / rate sensitivity (44 documents, 17%)
3. Unrealized HTM loss / duration risk (44 documents, 17%)
4. Credit loss provisioning (37 documents, 15%)
5. Commercial real estate / office (33 documents, 13%)
6. Capital adequacy / CET1 / regulatory (25 documents, 10%)
7. Technology sector / venture capital (18 documents, 7%)

The "deposits + liquidity" and "HTM losses" clusters being the two most prevalent topics reflects the structural design of the scenario: the 2022 rate hiking cycle generated both themes simultaneously, and they co-appear in distressed-bank filings. A BERTopic run on real EDGAR data would find these clusters empirically rather than from designed templates.

---

## 8. Survivorship-Bias Analysis

**The mechanical source of the bias**: After SIVB fails in March 2023 and FRC in May 2023, they stop filing. Any quarterly aggregation of surviving banks' sentiment will be missing their contributions, which were the most negative in the sample. The sector average must improve mechanically, regardless of any genuine recovery.

**The numbers**:
- Sector mean with all eight banks across the full period: **+0.015**
- Sector mean with only the six survivors: **+0.042**
- Survivorship bias: **+0.026 (+2.6 pp)**

The post-failure quarters (2023 Q1 onward) show identical "all" and "survivors" averages because the failed banks are no longer in either sample, the bias fully absorbs once the failures become historical. The bias is concentrated in the 2022 quarters, where SIVB and FRC were filing with deeply negative scores that pulled the sector average down relative to the survivors-only view.

**Implication for market analysis**: An analyst who constructs a regional-bank peer index that delists failed banks (standard practice in backtesting) will see a more positive trend in 2022–2024 than one who includes the failed-bank observations. The 2.6 pp inflation is not large enough to reverse a sector view, but it is large enough to shift a bank from "watch" to "hold" if evaluated on a curve relative to a survivorship-biased sector benchmark.

---

## 9. Disaggregated Recommendation

| Tier | Anomaly Signal | Gap Signal | Recommended Action |
|---|---|---|---|
| GSIB (JPM, BAC) | COVID-era flags only | Gap ~0.31, stable | **Hold**: resilient, no structural concern |
| Super-Regional (USB, RF) | COVID + isolated 2022 | Gap ~0.36, recovering | **Monitor**: elevated but declining post-cycle |
| Stressed survivors (KEY, ZION) | COVID + 2022–2023 | Gap 0.39–0.97, wide | **Caution**: warranted deeper review; survived but not comfortable |
| Failed (SIVB, FRC) | 2022 Q4 z < −1.71 | SIVB gap peaked 0.925 (2021 Q2); FRC 0.62 in 2022 Q4 | **Exit signal at 2022 Q4**: one quarter before failure |

---

## 10. Follow-Up Monitoring

1. **ZION and KEY quarterly**: If either shows a section gap above 0.80 or a combined z-score below −1.5 in consecutive quarters, escalate to fundamental review.
2. **Commercial real estate exposure**: The "CRE / office" topic cluster is the most likely source of the next stress cycle as 2022 originations refinance at higher rates in 2024–2026. Entities "commercial real estate" and "office" rising in risk-factor frequency is an early indicator.
3. **Survivorship-adjusted sector average**: Any peer-group analysis comparing bank sentiment should apply the +2.6 pp correction to account for the removed observations, or explicitly include the failed banks' last-filed scores.
4. **Real EDGAR run**: With `USE_FINBERT = True` and `USE_SYNTHETIC = False`, the pipeline re-runs on actual SEC filings. The section-gap and anomaly signals should be validated against the real FinBERT scores to confirm that the planted structure in the synthetic data reflects the actual filing language.

---

## Appendix: What Would Differ on Real EDGAR Data

| Aspect | Synthetic | Real EDGAR |
|---|---|---|
| Sentiment variance | Controlled (σ = 0.055 + noise) | Higher; legal boilerplate inflates neutral scores, dampening positive/negative extremes |
| FinBERT calibration | r = 0.91 mock ↔ true | True FinBERT probabilities; no planted ground truth |
| BERTopic topics | 7 designed clusters | Empirically discovered; may find unexpected themes (e.g., crypto, ESG) |
| Section parsing | Clean split by `section` column | Requires HTML extraction; section boundaries are approximate |
| Entity counts | Scaled to corpus size | Real entities from actual filing text; much richer geographic detail |
| Survivorship | Two known failures | Unknown future failures; analyst must track delisted/acquired entities |
| Domain shift | Templates match FinBERT domain | Legal boilerplate sections may score near-neutral; FinBERT accuracy ~77% on financial news vs. unknown on raw SEC filings |

The most significant difference is FinBERT's performance on raw SEC filings vs. the analyst reports and earnings transcripts it was trained on. Boilerplate risk-factor language ("there can be no assurance that...") may score neutral despite being semantically negative. Calibrating against human-annotated SEC filing segments before production use is recommended.
