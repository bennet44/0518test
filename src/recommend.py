"""Buy/sell recommendation scoring — a fund-manager-style composite score.

Combines momentum, risk-adjusted return, trend, and valuation into a single
cross-sectional z-score so candidates can be ranked against each other.
"""
import numpy as np
import pandas as pd

from . import data_loader as dl
from . import risk as risk_mod
from . import technical as ta

FACTOR_WEIGHTS = {
    "期間報酬率": 0.3,
    "Sharpe Ratio": 0.3,
    "趨勢(價格/SMA50)": 0.2,
    "估值(1/預估PE)": 0.2,
}


def _zscore(series: pd.Series) -> pd.Series:
    std = series.std()
    if pd.isna(std) or std == 0:
        return pd.Series(0.0, index=series.index)
    return (series - series.mean()) / std


def build_recommendation_table(tickers: list[str], period: str, risk_free_rate: float = 0.0) -> pd.DataFrame:
    rows = {}
    for t in tickers:
        df = dl.get_price_history(t, period=period)
        if df.empty or len(df) < 2:
            continue
        close = df["Close"]
        rets = risk_mod.daily_returns(close)
        sma50 = ta.sma(close, 50).iloc[-1]
        info = dl.get_company_info(t)
        pe = info.get("forwardPE") or info.get("trailingPE")
        rows[t] = {
            "期間報酬率": close.iloc[-1] / close.iloc[0] - 1,
            "Sharpe Ratio": risk_mod.sharpe_ratio(rets, risk_free_rate),
            "趨勢(價格/SMA50)": (close.iloc[-1] / sma50 - 1) if pd.notna(sma50) and sma50 else np.nan,
            "估值(1/預估PE)": (1 / pe) if pe and pe > 0 else np.nan,
            "RSI (14)": ta.rsi(close).iloc[-1],
        }
    table = pd.DataFrame(rows).T
    if table.empty:
        return table

    score = pd.Series(0.0, index=table.index)
    for factor, weight in FACTOR_WEIGHTS.items():
        score = score + _zscore(table[factor].astype(float)) * weight
    table["綜合評分"] = score
    return table.sort_values("綜合評分", ascending=False)


def top_buy_sell(table: pd.DataFrame, n: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the ranked table into disjoint buy/sell groups.

    Caps each side at len(table)//2 so a small candidate list never lets the
    same ticker appear in both the buy and sell lists.
    """
    if table.empty:
        return table, table
    sorted_desc = table.sort_values("綜合評分", ascending=False)
    total = len(sorted_desc)
    if total == 1:
        return sorted_desc, sorted_desc.iloc[0:0]
    n_eff = min(n, total // 2)
    buy = sorted_desc.head(n_eff)
    sell = sorted_desc.tail(n_eff).iloc[::-1]
    return buy, sell
