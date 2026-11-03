# Validation

I checked the extraction, backend failure handling, provenance and annotation requirements. All 25 Python tests passed, and a real FinBERT smoke check processed multiple chunks without truncation.

I downloaded nine predefined annual filings and scored nine usable sections, processing all 327,746 content tokens. Source review caught inline references that contaminated section boundaries; regression tests now cover that failure. A cached rerun checked the model revision, exact section text and saved score hashes before reusing results.

## Scope

The authentic sentiment outputs remain preliminary. Missing or incorporated-by-reference sections are explicit, and only BAC has paired MD&A/Risk Factors observations in this primary-document extraction. Independent human labels and adjudication are still required. The historical mock sentiment, entity and topic outputs are separate from this authentic sentiment run. See [REPORT.md](REPORT.md), [the study protocol](docs/authentic-study.md) and `outputs/authentic/manifest.json` for the evidence and remaining evaluation.
