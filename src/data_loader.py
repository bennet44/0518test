"""Cached data access layer around yfinance."""
import pandas as pd
import streamlit as st
import yfinance as yf


@st.cache_data(ttl=3600, show_spinner=False)
def get_price_history(ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    try:
        df = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=True)
    except Exception:
        return pd.DataFrame()
    if df.empty:
        return df
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def get_multi_close(tickers: list[str], period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    closes = {}
    for t in tickers:
        df = get_price_history(t, period=period, interval=interval)
        if not df.empty:
            closes[t] = df["Close"]
    return pd.DataFrame(closes).dropna(how="all")


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def get_company_info(ticker: str) -> dict:
    try:
        return yf.Ticker(ticker).get_info()
    except Exception:
        return {}


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def get_fundamentals_table(tickers: list[str]) -> pd.DataFrame:
    fields = {
        "shortName": "公司名稱",
        "sector": "產業",
        "marketCap": "市值",
        "trailingPE": "P/E (TTM)",
        "forwardPE": "預估 P/E",
        "priceToBook": "P/B",
        "trailingEps": "EPS (TTM)",
        "revenueGrowth": "營收成長率",
        "earningsGrowth": "盈餘成長率",
        "profitMargins": "淨利率",
        "returnOnEquity": "ROE",
        "dividendYield": "股息率",
        "beta": "Beta",
        "fiftyTwoWeekHigh": "52週高",
        "fiftyTwoWeekLow": "52週低",
    }
    rows = {}
    for t in tickers:
        info = get_company_info(t)
        rows[t] = {label: info.get(key) for key, label in fields.items()}
    return pd.DataFrame(rows).T
