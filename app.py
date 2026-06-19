"""US Stock Analyst Dashboard — Streamlit app."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import data_loader as dl
from src import recommend
from src import risk
from src import technical as ta

st.set_page_config(page_title="美股分析師看板", layout="wide")

PERIOD_OPTIONS = {
    "1個月": "1mo", "3個月": "3mo", "6個月": "6mo",
    "1年": "1y", "2年": "2y", "5年": "5y",
}

with st.sidebar:
    st.title("📊 美股分析師看板")
    tickers_input = st.text_input("股票代號（逗號分隔）", value="AAPL, MSFT, NVDA")
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
    period_label = st.selectbox("時間範圍", list(PERIOD_OPTIONS.keys()), index=3)
    period = PERIOD_OPTIONS[period_label]
    risk_free_rate = st.number_input("無風險利率（年化，%）", value=4.0, step=0.1) / 100
    top_n = st.selectbox("建議買賣標的數量 (Top N)", [1, 5, 10, 15], index=2)

if not tickers:
    st.warning("請至少輸入一個股票代號。")
    st.stop()

primary = tickers[0]

tab_price, tab_fundamentals, tab_compare, tab_risk, tab_reco = st.tabs(
    ["📈 價格與技術指標", "🧾 基本面財務", "🔗 多股比較與關聯", "⚖️ 風險與統計", "💡 買賣建議"]
)

# ---------- Tab 1: Price & Technical ----------
with tab_price:
    st.subheader(f"{primary} 價格與技術指標")
    df = dl.get_price_history(primary, period=period)
    if df.empty:
        st.error(f"找不到 {primary} 的資料，請確認代號是否正確。")
    else:
        close = df["Close"]
        sma20, sma50 = ta.sma(close, 20), ta.sma(close, 50)
        bb = ta.bollinger_bands(close)

        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
            name=primary,
        ))
        fig.add_trace(go.Scatter(x=df.index, y=sma20, name="SMA20", line=dict(width=1)))
        fig.add_trace(go.Scatter(x=df.index, y=sma50, name="SMA50", line=dict(width=1)))
        fig.add_trace(go.Scatter(x=df.index, y=bb["upper"], name="Bollinger Upper",
                                  line=dict(width=1, dash="dot"), opacity=0.5))
        fig.add_trace(go.Scatter(x=df.index, y=bb["lower"], name="Bollinger Lower",
                                  line=dict(width=1, dash="dot"), opacity=0.5,
                                  fill="tonexty"))
        fig.update_layout(height=500, xaxis_rangeslider_visible=False,
                           margin=dict(t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

        vol_fig = go.Figure(go.Bar(x=df.index, y=df["Volume"], name="Volume"))
        vol_fig.update_layout(height=180, margin=dict(t=10, b=10), title="成交量")
        st.plotly_chart(vol_fig, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            rsi_series = ta.rsi(close)
            rsi_fig = go.Figure(go.Scatter(x=df.index, y=rsi_series, name="RSI"))
            rsi_fig.add_hline(y=70, line_dash="dash", line_color="red")
            rsi_fig.add_hline(y=30, line_dash="dash", line_color="green")
            rsi_fig.update_layout(height=250, title="RSI (14)", margin=dict(t=30, b=10))
            st.plotly_chart(rsi_fig, use_container_width=True)
        with col2:
            macd_df = ta.macd(close)
            macd_fig = go.Figure()
            macd_fig.add_trace(go.Scatter(x=df.index, y=macd_df["macd"], name="MACD"))
            macd_fig.add_trace(go.Scatter(x=df.index, y=macd_df["signal"], name="Signal"))
            macd_fig.add_trace(go.Bar(x=df.index, y=macd_df["hist"], name="Histogram"))
            macd_fig.update_layout(height=250, title="MACD", margin=dict(t=30, b=10))
            st.plotly_chart(macd_fig, use_container_width=True)

        latest = close.iloc[-1]
        prev = close.iloc[-2] if len(close) > 1 else latest
        st.metric(f"{primary} 最新收盤價", f"${latest:,.2f}",
                   f"{(latest / prev - 1) * 100:.2f}%")

# ---------- Tab 2: Fundamentals ----------
with tab_fundamentals:
    st.subheader("基本面財務數據比較")
    fdf = dl.get_fundamentals_table(tickers)
    if fdf.empty:
        st.warning("無法取得基本面資料。")
    else:
        display = fdf.copy()
        if "市值" in display:
            display["市值"] = display["市值"].apply(
                lambda v: f"${v / 1e9:,.1f}B" if pd.notnull(v) else None)
        for pct_col in ["營收成長率", "盈餘成長率", "淨利率", "ROE", "股息率"]:
            if pct_col in display:
                display[pct_col] = display[pct_col].apply(
                    lambda v: f"{v * 100:.2f}%" if pd.notnull(v) else None)
        st.dataframe(display, use_container_width=True)

        numeric_cols = ["P/E (TTM)", "預估 P/E", "P/B", "Beta"]
        plot_df = fdf[numeric_cols].apply(pd.to_numeric, errors="coerce")
        if plot_df.notna().any().any():
            metric = st.selectbox("比較指標", numeric_cols)
            bar_fig = go.Figure(go.Bar(x=plot_df.index, y=plot_df[metric]))
            bar_fig.update_layout(height=350, title=f"{metric} 比較", margin=dict(t=40, b=10))
            st.plotly_chart(bar_fig, use_container_width=True)

# ---------- Tab 3: Multi-stock comparison & correlation ----------
with tab_compare:
    st.subheader("多股票報酬比較與相關性")
    close_df = dl.get_multi_close(tickers, period=period)
    if close_df.empty or len(tickers) < 2:
        st.info("請輸入至少兩個股票代號以進行比較。")
    else:
        normalized = close_df / close_df.iloc[0] * 100
        norm_fig = go.Figure()
        for t in normalized.columns:
            norm_fig.add_trace(go.Scatter(x=normalized.index, y=normalized[t], name=t))
        norm_fig.update_layout(height=400, title="累積報酬比較（基準=100）",
                                margin=dict(t=40, b=10))
        st.plotly_chart(norm_fig, use_container_width=True)

        corr = risk.correlation_matrix(close_df)
        heat_fig = go.Figure(go.Heatmap(
            z=corr.values, x=corr.columns, y=corr.index,
            colorscale="RdBu", zmid=0, text=corr.round(2).values,
            texttemplate="%{text}",
        ))
        heat_fig.update_layout(height=400, title="日報酬相關係數矩陣", margin=dict(t=40, b=10))
        st.plotly_chart(heat_fig, use_container_width=True)

# ---------- Tab 4: Risk & statistics ----------
with tab_risk:
    st.subheader("風險與統計分析")
    rows = {}
    for t in tickers:
        df_t = dl.get_price_history(t, period=period)
        if not df_t.empty:
            rows[t] = risk.risk_summary(df_t["Close"], risk_free_rate)
    if rows:
        summary_df = pd.DataFrame(rows).T
        fmt = summary_df.copy()
        for col in ["年化報酬率", "年化波動率", "最大回撤", "VaR (95%, 日)"]:
            fmt[col] = fmt[col].apply(lambda v: f"{v * 100:.2f}%" if pd.notnull(v) else None)
        fmt["Sharpe Ratio"] = fmt["Sharpe Ratio"].apply(
            lambda v: f"{v:.2f}" if pd.notnull(v) else None)
        st.dataframe(fmt, use_container_width=True)

        st.markdown("#### 日報酬率分布")
        hist_fig = go.Figure()
        for t in tickers:
            df_t = dl.get_price_history(t, period=period)
            if not df_t.empty:
                rets = risk.daily_returns(df_t["Close"]) * 100
                hist_fig.add_trace(go.Histogram(x=rets, name=t, opacity=0.6, nbinsx=60))
        hist_fig.update_layout(barmode="overlay", height=350,
                                xaxis_title="日報酬率 (%)", margin=dict(t=20, b=10))
        st.plotly_chart(hist_fig, use_container_width=True)
    else:
        st.warning("無可用資料以計算風險指標。")

# ---------- Tab 5: Buy/sell recommendations ----------
with tab_reco:
    st.subheader("基金經理人觀點：建議買入 / 賣出")
    st.caption(
        "綜合「期間報酬率」「Sharpe Ratio」「價格趨勢（價格 / SMA50）」「估值（1/預估PE）」"
        "四項因子計算組內相對評分，僅反映目前清單內標的之相對排序，非投資建議。"
    )
    reco_table = recommend.build_recommendation_table(tickers, period, risk_free_rate)
    if reco_table.empty:
        st.warning("無足夠資料產生建議，請確認股票代號或時間範圍。")
    else:
        buy_df, sell_df = recommend.top_buy_sell(reco_table, top_n)

        def _format_reco(df: pd.DataFrame) -> pd.DataFrame:
            fmt = df.copy()
            for col in ["期間報酬率", "趨勢(價格/SMA50)"]:
                fmt[col] = fmt[col].apply(lambda v: f"{v * 100:.2f}%" if pd.notnull(v) else None)
            for col in ["Sharpe Ratio", "估值(1/預估PE)", "RSI (14)", "綜合評分"]:
                fmt[col] = fmt[col].apply(lambda v: f"{v:.2f}" if pd.notnull(v) else None)
            return fmt

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"#### 🟢 建議買入 Top {len(buy_df)}")
            st.dataframe(_format_reco(buy_df), use_container_width=True)
        with col2:
            st.markdown(f"#### 🔴 建議賣出 Top {len(sell_df)}")
            st.dataframe(_format_reco(sell_df), use_container_width=True)

        if len(reco_table) < top_n:
            st.info(f"目前清單僅有 {len(reco_table)} 檔標的，少於選擇的 Top {top_n}，已顯示全部可用標的。")
