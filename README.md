# Financial NLP: authentic sections and synthetic workflow examples

Real FinBERT executed on nine extracted sections from nine predefined 2022-2024 annual SEC filings, with all 327,746 content tokens processed and zero truncation. Primary-document extraction leaves nine requested sections missing or incorporated by reference. Human sentiment labels are still required, so model outputs are preliminary.

See [REPORT.md](REPORT.md) for current evidence, [protocol and rubric](docs/authentic-study.md), [manifest](outputs/authentic/manifest.json) and [validation](VALIDATION.md). Install `requirements-authentic.txt` in an isolated Python 3.14 environment, set SEC_CONTACT locally, and run `scripts/run_authentic.py`. The earlier generated scenario and mock entity/topic outputs remain separate from this authentic sentiment run.
