"""A requested real backend must never silently become a mock result."""
import pandas as pd
import pytest
from src.nlp_ei import sentiment


def test_real_backend_failure_propagates(monkeypatch):
    def unavailable(*args, **kwargs):
        raise RuntimeError("model unavailable")
    monkeypatch.setattr(sentiment, "score_with_finbert", unavailable)
    with pytest.raises(RuntimeError, match="model unavailable"):
        sentiment.run_sentiment(pd.DataFrame({"text": ["filing text"]}), use_finbert=True)


def test_mock_backend_is_explicit():
    result = sentiment.run_sentiment(
        pd.DataFrame({"text": ["synthetic text"], "true_sentiment": [0.2]}),
        use_finbert=False,
    )
    assert result.attrs["requested_sentiment_backend"] == "mock"
    assert result.attrs["executed_sentiment_backend"] == "mock"
