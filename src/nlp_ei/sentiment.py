"""FinBERT sentiment analysis on filing text."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _chunk_text(text: str, max_chars: int = 512 * 4) -> list[str]:
    """Split long text into sentence-ish chunks for FinBERT (512-token limit)."""
    sentences = [s.strip() for s in text.replace("\n", " ").split(".") if len(s.strip()) > 20]
    chunks, current, current_len = [], [], 0
    for s in sentences:
        if current_len + len(s) > max_chars and current:
            chunks.append(". ".join(current) + ".")
            current, current_len = [s], len(s)
        else:
            current.append(s)
            current_len += len(s)
    if current:
        chunks.append(". ".join(current) + ".")
    return chunks or [text[:max_chars]]


def score_with_finbert(
    texts: list[str],
    model_name: str = "ProsusAI/finbert",
    batch_size: int = 16,
    device: str = "cpu",
) -> list[dict]:
    """
    Run FinBERT on a list of texts and return per-text sentiment dicts.

    Returns: list of {positive, negative, neutral, label, score}
    where score = positive − negative (net sentiment).

    Long texts (>512 tokens) are split into chunks and scores are averaged.
    Averaging is a pragmatic choice; an alternative is first-N-sentences scoring
    to avoid dilution by boilerplate legal language.
    """
    try:
        from transformers import pipeline
        pipe = pipeline(
            "text-classification",
            model=model_name,
            tokenizer=model_name,
            device=device,
            top_k=None,
            max_length=512,
            truncation=True,
        )
    except ImportError:
        raise ImportError("pip install transformers torch")

    results = []
    for text in texts:
        chunks = _chunk_text(text)
        chunk_scores = []
        for batch_start in range(0, len(chunks), batch_size):
            batch = chunks[batch_start: batch_start + batch_size]
            outputs = pipe(batch)
            for item in outputs:
                probs = {d["label"].lower(): d["score"] for d in item}
                chunk_scores.append(probs)

        avg = {k: np.mean([c.get(k, 0) for c in chunk_scores]) for k in ["positive", "negative", "neutral"]}
        avg["score"] = avg["positive"] - avg["negative"]
        avg["label"] = max(["positive", "negative", "neutral"], key=lambda k: avg[k])
        results.append(avg)

    return results


def score_mock(
    texts: list[str],
    true_sentiments: list[float] | None = None,
    seed: int = 99,
) -> list[dict]:
    """
    Deterministic mock scorer for when FinBERT is not installed.

    If true_sentiments (values in [−1, 1]) are provided, produces scores that
    correlate with the planted ground truth, preserving the latent structure
    in the synthetic corpus. Without true_sentiments, draws random values.

    Returns scores in the same format as score_with_finbert.
    """
    rng = np.random.default_rng(seed)
    results = []

    for i, text in enumerate(texts):
        if true_sentiments is not None:
            ts = float(np.clip(true_sentiments[i], -1.0, 1.0))
            # Map [-1, 1] → pos/neg centres
            # ts=+1  → pos≈0.80, neg≈0.05
            # ts=0   → pos≈0.43, neg≈0.43
            # ts=−1  → pos≈0.05, neg≈0.80
            pos_center = 0.425 + 0.375 * ts
            neg_center = 0.425 - 0.375 * ts
        else:
            pos_center = 0.40
            neg_center = 0.35

        pos = float(np.clip(rng.normal(pos_center, 0.08), 0.02, 0.96))
        neg = float(np.clip(rng.normal(neg_center, 0.08), 0.02, 0.96))

        total = pos + neg
        if total > 0.95:
            pos = pos / total * 0.95
            neg = neg / total * 0.95
        neu = float(np.clip(1.0 - pos - neg, 0.0, 1.0))

        results.append({
            "positive": round(pos, 4),
            "negative": round(neg, 4),
            "neutral": round(neu, 4),
            "score": round(pos - neg, 4),
            "label": "positive" if pos > neg else "negative",
        })

    return results


def run_sentiment(
    corpus_df: pd.DataFrame,
    use_finbert: bool = True,
    model_name: str = "ProsusAI/finbert",
    batch_size: int = 16,
) -> pd.DataFrame:
    """Apply sentiment scoring to all rows of corpus_df.

    When use_finbert=False and corpus_df contains a 'true_sentiment' column,
    the mock scorer uses that as a correlated signal so the latent structure
    planted in the synthetic corpus is preserved through the pipeline.
    """
    texts = corpus_df["text"].tolist()

    if use_finbert:
        try:
            scores = score_with_finbert(texts, model_name=model_name, batch_size=batch_size)
        except (ImportError, Exception) as e:
            print(f"FinBERT unavailable ({e}), falling back to mock scorer")
            true_sents = (
                corpus_df["true_sentiment"].tolist()
                if "true_sentiment" in corpus_df.columns else None
            )
            scores = score_mock(texts, true_sentiments=true_sents)
    else:
        true_sents = (
            corpus_df["true_sentiment"].tolist()
            if "true_sentiment" in corpus_df.columns else None
        )
        scores = score_mock(texts, true_sentiments=true_sents)

    sentiment_df = pd.DataFrame(scores)
    return pd.concat([corpus_df.reset_index(drop=True), sentiment_df], axis=1)
