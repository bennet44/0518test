"""Stock universe helpers — currently just the S&P 500 constituent list."""
import pandas as pd
import streamlit as st

_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

# Small, definitely-correct fallback used only if the live Wikipedia fetch
# fails (e.g. no network access or the page structure changed), so "ALL"
# mode still returns something usable.
_FALLBACK_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "BRK-B", "TSLA", "LLY", "AVGO",
    "JPM", "V", "UNH", "XOM", "MA", "JNJ", "PG", "HD", "MRK", "COST",
    "ABBV", "CVX", "CRM", "NFLX", "AMD", "PEP", "KO", "WMT", "BAC", "TMO",
    "ADBE", "MCD", "CSCO", "ABT", "ORCL", "ACN", "LIN", "DHR", "WFC", "DIS",
    "TXN", "PM", "INTU", "VZ", "CMCSA", "IBM", "NOW", "CAT", "GE", "UNP",
]


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def get_sp500_tickers() -> list[str]:
    """Live S&P 500 ticker list scraped from Wikipedia, cached for a day.

    Falls back to a short list of well-known constituents if the fetch
    fails, so "ALL" mode keeps working without network access to Wikipedia.
    """
    try:
        tables = pd.read_html(_WIKI_URL)
        symbols = (
            tables[0]["Symbol"].astype(str).str.strip().str.replace(".", "-", regex=False)
        )
        tickers = sorted(set(symbols.tolist()))
        if len(tickers) >= 400:
            return tickers
    except Exception:
        pass
    return _FALLBACK_TICKERS
