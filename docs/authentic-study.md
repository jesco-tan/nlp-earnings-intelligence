# Authentic filing study protocol

Frozen panel: JPM, BAC and WFC, annual 10-K reporting periods 2022, 2023 and 2024. Compare MD&A and Risk Factors only. No quarterly equivalence, anomaly-alert or failure-prediction claim. The three annual observations per series are insufficient for an independent earlier-period threshold and later-period alert evaluation.

Run `.venv/Scripts/python.exe scripts/run_authentic.py` with SEC_CONTACT set to the owner-approved real email. The contact is sent in the SEC User-Agent and is never persisted in the manifest. Requests are sequential, at most four per second; failures stop the run. Cached original bytes are hashed and kept under ignored data directories. Accession, URL, report date and acceptance time travel with every section.

Extraction chooses the longest valid Item 1A to 1B and Item 7 to 7A span between short block headings; inline cross-references are excluded. It explicitly marks missing sections and requires manual boundary review before acceptance. Review beginning/end text for every section and compare at least one full section per bank against its source. Heading variation and inline markup can defeat this heuristic; missing or contaminated extraction must be repaired before interpretation.

FinBERT model and tokenizer share a pinned immutable revision. Full tokenizer output is partitioned into nonoverlapping chunks within model capacity. All content tokens are scored, with zero truncation and token-weighted aggregation. This is a pragmatic section summary, not an independently validated document-level sentiment measure. Do not replace failures with mock scores.

## Independent annotation rubric

Two human annotators independently label the held-out sentences without seeing model outputs. Positive: an explicit favourable financial condition, performance or outlook. Negative: an explicit adverse financial condition, deterioration or material risk. Neutral: descriptive facts, procedures or no directional financial assessment. Conditional boilerplate alone is neutral unless it states a concrete adverse condition. Flag ambiguous/mixed sentences for adjudication; agree one label before scoring. Save both original labels and the adjudicated label. Do not use model-generated labels as gold.

Five sentences are uniformly sampled per filing/section using seed 42, up to 90 total. This is a small within-corpus evaluation; report class support and uncertainty rather than asserting general financial-domain accuracy. A human must fill annotator_a, annotator_b and adjudicated_label. Model predictions remain separate until labels are frozen. Report per-class precision/recall/F1, confusion matrix and annotator disagreement. A sentence sample is not a bank-failure study.

Model reference: https://huggingface.co/ProsusAI/finbert. SEC access rules: https://www.sec.gov/about/developer-resources.

FinBERT attribution: Dogu Araci (2019), FinBERT: Financial Sentiment Analysis with Pre-trained Language Models. The current model card attributes weights to MIH AI B.V. under CC BY-NC-SA 4.0. Model weights are kept in the ignored local cache.
