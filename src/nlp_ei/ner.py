"""Named Entity Recognition on filing text using spaCy."""

from __future__ import annotations

import pandas as pd
from collections import Counter


def extract_entities(
    corpus_df: pd.DataFrame,
    spacy_model: str = "en_core_web_sm",
    entity_types: list[str] | None = None,
) -> pd.DataFrame:
    """
    Run spaCy NER on each row's text and return a flat frequency DataFrame.

    Returns: DataFrame with columns: entity, label, count
    """
    if entity_types is None:
        entity_types = ["ORG", "GPE", "PRODUCT", "MONEY", "PERCENT", "DATE"]

    try:
        import spacy
        nlp = spacy.load(spacy_model)
    except (ImportError, OSError):
        print(f"spaCy model {spacy_model!r} not found, returning banking mock entities")
        return _mock_entities(corpus_df)

    records = []
    for _, row in corpus_df.iterrows():
        doc = nlp(row["text"][:10_000])
        for ent in doc.ents:
            if ent.label_ in entity_types:
                records.append({
                    "ticker": row["ticker"],
                    "filing_date": row.get("filing_date"),
                    "section": row.get("section"),
                    "entity": ent.text.strip(),
                    "label": ent.label_,
                })

    if not records:
        return pd.DataFrame(columns=["entity", "label", "count"])

    df = pd.DataFrame(records)
    freq = df.groupby(["entity", "label"]).size().reset_index(name="count")
    return freq.sort_values("count", ascending=False).reset_index(drop=True)


def _mock_entities(corpus_df: pd.DataFrame) -> pd.DataFrame:
    """
    Banking-domain mock entity table for Tier 1 runs without spaCy.
    Entities reflect the regional bank stress scenario 2020–2024.
    """
    mock = [
        # Geographic risk concentration
        ("United States",             "GPE",     312),
        ("California",                "GPE",     187),
        ("New York",                  "GPE",     165),
        ("China",                     "GPE",      98),
        ("Europe",                    "GPE",      87),
        ("Texas",                     "GPE",      72),
        # Regulatory / institutional bodies
        ("Federal Reserve",           "ORG",     284),
        ("FDIC",                      "ORG",     156),
        ("Federal Home Loan Bank",    "ORG",      89),
        ("Office of the Comptroller of the Currency", "ORG", 64),
        ("U.S. Treasury",             "ORG",      58),
        ("Basel Committee",           "ORG",      41),
        # Products / instruments
        ("commercial real estate",    "PRODUCT",  203),
        ("held-to-maturity",          "PRODUCT",  178),
        ("Paycheck Protection Program","PRODUCT",  134),
        ("certificates of deposit",   "PRODUCT",   88),
        ("interest rate swaps",       "PRODUCT",   63),
        # Financial magnitudes
        ("$1 billion",                "MONEY",     96),
        ("$500 million",              "MONEY",     84),
        ("$2 billion",                "MONEY",     71),
        ("$250 million",              "MONEY",     58),
        ("10%",                       "PERCENT",  142),
        ("5%",                        "PERCENT",  118),
        ("1.5%",                      "PERCENT",   89),
        ("525 basis points",          "PERCENT",   62),
    ]
    df = pd.DataFrame(mock, columns=["entity", "label", "count"])

    # Scale counts proportionally to corpus size
    n = len(corpus_df)
    scale = max(1.0, n / 120)
    df["count"] = (df["count"] * scale).astype(int)
    return df.sort_values("count", ascending=False).reset_index(drop=True)


def top_entities_by_type(entity_df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Return top N entities per NER label type."""
    return (
        entity_df
        .groupby("label", group_keys=False)
        .apply(lambda g: g.nlargest(n, "count"))
        .reset_index(drop=True)
    )
