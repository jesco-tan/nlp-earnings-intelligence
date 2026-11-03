# Financial NLP workflow: a synthetic bank-disclosure scenario

Implemented a financial-text workflow demonstrated on 252 generated filing-style sections across eight bank labels. The saved run uses correlated mock sentiment, mock entities and mock topics, with section-gap analysis, retrospective z-scores and a historical cohort comparison. Optional FinBERT, spaCy and BERTopic adapters are implemented; real SEC/FinBERT execution is not verified.

I built a workflow for comparing operating and risk language, aggregating sentiment and investigating unusual changes within an entity. The published demonstration uses synthetic filing-style text and mock NLP outputs. It illustrates the analysis machinery; it does not show that a model anticipated actual bank failures.

## What the published run contains

## Read the results

[REPORT.md](REPORT.md) gives the supported results, assumptions and limitations. [report.html](outputs/report.html) contains the same interpretation with embedded figures. The [walkthrough notebook](notebooks/nlp_walkthrough.ipynb) retains numerical evidence and corrected narrative. Modified code cells have cleared execution state until rerun. Old wording and reports are available through Git history.

No real users, customers, bank-disclosure measurements or policy effects are represented by this saved run.

## Reproduce or inspect

Render the current report without dependencies or analytical execution:

```powershell
python scripts/generate_html_report.py
```

Run the synthetic notebook in an isolated environment (this overwrites analytical outputs):

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
# Install notebook tooling inside the same virtual environment if needed.
.venv/Scripts/python.exe -m pip install jupyter
.venv/Scripts/python.exe -m jupyter notebook notebooks/nlp_walkthrough.ipynb
.venv/Scripts/python.exe -m pytest tests -q
```

The analytical requirements pin an older scientific stack. Use a compatible Python 3.11 virtual environment for the full notebook; Python 3.14 can render the report using the standard library without installing that stack.

## Evidence and implementation limits

Financial-text workflow demonstrated on synthetic bank-disclosure scenarios, with optional FinBERT, spaCy and BERTopic adapters. Published results use mock NLP outputs.

See [REPORT.md](REPORT.md) for method definitions, saved-run provenance and remaining analytical limitations.
