"""BERTopic topic modelling on filing text."""

from __future__ import annotations

import numpy as np
import pandas as pd


def fit_bertopic(
    texts: list[str],
    min_topic_size: int = 5,
    n_gram_range: tuple[int, int] = (1, 2),
    seed: int = 42,
) -> tuple:
    """
    Fit a BERTopic model and return (model, topics, probs).

    Requires: bertopic sentence-transformers umap-learn hdbscan
    BERTopic assigns topic -1 to outliers (HDBSCAN noise); expect 10–30% of docs
    labelled -1, which should be excluded from topic-level analysis.
    """
    try:
        from bertopic import BERTopic
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError("pip install bertopic sentence-transformers umap-learn hdbscan")

    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    topic_model = BERTopic(
        embedding_model=embedding_model,
        min_topic_size=min_topic_size,
        n_gram_range=n_gram_range,
        calculate_probabilities=True,
        verbose=False,
        seed_topic_list=None,
    )
    topics, probs = topic_model.fit_transform(texts)
    return topic_model, topics, probs


def get_topic_info(topic_model) -> pd.DataFrame:
    """Return topic labels and top keywords."""
    return topic_model.get_topic_info()


def mock_topics(texts: list[str]) -> pd.DataFrame:
    """
    Return a banking-domain mock topic table when BERTopic dependencies are unavailable.

    Topics reflect the latent themes in the regional bank stress scenario:
    - Deposit stability & liquidity stress
    - Net interest margin & rate sensitivity
    - Unrealized HTM securities losses
    - Credit quality & loan loss provisioning
    - Capital adequacy & regulatory compliance
    """
    n = len(texts)
    rng = np.random.default_rng(0)

    topic_labels = [
        "deposits, liquidity, outflows, uninsured",
        "net interest margin, rate, asset-sensitive, repricing",
        "held-to-maturity, unrealized loss, securities, duration",
        "credit loss, provision, non-performing, charge-offs",
        "capital, CET1, regulatory, stress test",
        "commercial real estate, office, vacancy, refinancing",
        "technology sector, venture capital, startup, SVB",
    ]
    # Weight distribution: more docs on rate/HTM/deposit themes (2022 dominates)
    weights = [0.20, 0.22, 0.18, 0.15, 0.10, 0.10, 0.05]
    topic_ids = rng.choice(len(topic_labels), size=n, p=weights)

    return pd.DataFrame({
        "text_idx": range(n),
        "topic_id": topic_ids,
        "topic_label": [topic_labels[i] for i in topic_ids],
    })
