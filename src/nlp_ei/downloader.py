"""Download SEC EDGAR filings via sec-edgar-downloader."""

from __future__ import annotations

from pathlib import Path
import pandas as pd


def download_filings(
    tickers: list[str],
    form_types: list[str],
    start_date: str,
    end_date: str,
    output_dir: Path | str = "data/filings",
) -> None:
    """
    Download 10-K and 10-Q filings for each ticker using sec-edgar-downloader.

    Requires: pip install sec-edgar-downloader
    """
    try:
        from sec_edgar_downloader import Downloader
    except ImportError:
        raise ImportError("pip install sec-edgar-downloader")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dl = Downloader("MyCompany", "myemail@example.com", output_dir)

    for ticker in tickers:
        for form in form_types:
            try:
                dl.get(form, ticker, after=start_date, before=end_date)
                print(f"  Downloaded {form} for {ticker}")
            except Exception as e:
                print(f"  WARN: {ticker} {form}: {e}")


def load_filing_texts(
    filings_dir: Path | str,
    tickers: list[str],
    sections: list[str] | None = None,
) -> pd.DataFrame:
    """
    Walk filings_dir, find primary document HTML/TXT, return DataFrame with
    columns: ticker, filing_date, form_type, section, text.
    """
    import re
    from bs4 import BeautifulSoup

    filings_dir = Path(filings_dir)
    records = []

    for ticker in tickers:
        ticker_dir = filings_dir / ticker.upper()
        if not ticker_dir.exists():
            continue
        for form_dir in ticker_dir.iterdir():
            for filing_dir in form_dir.iterdir():
                for doc in filing_dir.glob("*.htm"):
                    try:
                        soup = BeautifulSoup(doc.read_text(encoding="utf-8", errors="ignore"), "html.parser")
                        full_text = soup.get_text(separator=" ", strip=True)
                        records.append({
                            "ticker": ticker,
                            "form_type": form_dir.name,
                            "filing_date": None,
                            "section": "full",
                            "text": full_text[:50_000],   # cap at 50K chars
                        })
                    except Exception:
                        pass

    return pd.DataFrame(records)
