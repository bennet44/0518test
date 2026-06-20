"""Stock universe helpers — S&P 500 constituents and a high-volume watchlist."""
import pandas as pd
import streamlit as st

from . import data_loader as dl

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

# Candidate pool of historically high-volume US tickers (large caps, popular
# retail/momentum names, leveraged ETFs) used to derive a "top N by recent
# volume" sub-universe. This is a heuristic watchlist, not a live market-wide
# volume screener.
_HIGH_VOLUME_CANDIDATES = [
    "AAPL", "TSLA", "NVDA", "AMD", "AMZN", "META", "MSFT", "GOOGL", "NFLX", "BAC",
    "F", "T", "INTC", "PFE", "NIO", "SOFI", "PLTR", "RIVN", "LCID", "AAL",
    "CCL", "PLUG", "SNAP", "UBER", "PYPL", "XOM", "WBD", "KVUE", "VALE", "ITUB",
    "SIRI", "GRAB", "MARA", "RIOT", "COIN", "SOXL", "TQQQ", "SQQQ", "SPY", "QQQ",
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


@st.cache_data(ttl=3600, show_spinner=False)
def get_top_volume_tickers(n: int = 30) -> list[str]:
    """Rank a curated watchlist of typically-liquid tickers by recent average
    daily volume (last 10 trading days) and return the top n symbols.
    """
    volumes = {}
    for t in _HIGH_VOLUME_CANDIDATES:
        df = dl.get_price_history(t, period="1mo")
        if not df.empty:
            volumes[t] = df["Volume"].tail(10).mean()
    ranked = sorted(volumes, key=volumes.get, reverse=True)
    return ranked[:n]

