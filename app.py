"""EGX stock ranking dashboard.

Main page : all stocks ranked from best to worst (click a row to open its details).
Detail page: chart + indicators + rank history for the selected stock.
The selected stock lives in the URL (?ticker=XXX), so a detail page can be bookmarked or shared.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import dashboard_data as dd

st.set_page_config(page_title="EGX Stock Ranking", page_icon="📈", layout="wide")


@st.cache_data(ttl=600, show_spinner=False)
def get_data(signature):
    # `signature` (file modification times) is only the cache key: new files -> reload.
    return dd.load_all()


try:
    data = get_data(dd.files_signature())
except FileNotFoundError as err:
    st.title("📈 EGX Stock Ranking")
    st.error(f"Data files are missing: {err}. Run the 'Daily EGX update' workflow on GitHub first.")
    st.stop()

pred, hist, scores = data["predictions"], data["history"], data["scores"]
report, status = data["report"], data["status"]
backtest = data["backtest"]
asof = pd.Timestamp(report.get("asof") or pred["Date"].max())


def fmt(value, spec="{:.2f}", suffix=""):
    if value is None or pd.isna(value):
        return "-"
    return spec.format(value) + suffix


def rating_label(rating):
    return dd.RATING_EMOJI.get(rating, "") + " " + str(rating)


# ----------------------------------------------------------------------------- header
def render_header():
    st.title("📈 EGX Stock Ranking")
    updated = status.get("updated_at_cairo") or report.get("generated_at") or "-"
    st.caption(f"Prices as of **{asof.date()}** · last update **{updated}** (Cairo time)")
    for msg in dd.build_warnings(status, report, asof):
        st.warning(msg)


def render_model_card():
    with st.expander("How to read this, and how the model did in testing"):
        st.markdown(
            "- **Score** is the model's output. It is used only to *rank* stocks against each other "
            "over a 10-trading-day horizon.\n"
            "- **Rating** is the stock's position in today's ranking: top 20% = Very Strong, down to "
            "bottom 20% = Very Weak. It is *relative*, so some stocks are always Very Strong, even in a falling market.\n"
            "- This is a model output for education, **not investment advice**. Past test results do not guarantee future results."
        )
        ratings = report.get("ratings")
        if ratings:
            period = report.get("test_period") or ["?", "?"]
            st.markdown(f"**Test period:** {period[0]} to {period[1]} · "
                        f"market average 10-day return: {fmt(report.get('market_avg_10d_return'), '{:+.2f}', '%')}")
            table = pd.DataFrame(ratings).rename(columns={
                "rating": "Rating", "n": "Observations", "avg_10d_return": "Avg 10d return %",
                "excess_vs_market": "Excess vs market %", "pct_up": "% that went up"})
            st.dataframe(table, hide_index=True)
            c1, c2, c3 = st.columns(3)
            c1.metric("Top-10 daily avg 10d return", fmt(report.get("top10_daily_avg_return"), "{:+.2f}", "%"))
            c2.metric("Bottom-10 daily avg 10d return", fmt(report.get("bottom10_daily_avg_return"), "{:+.2f}", "%"))
            c3.metric("Rank correlation (IC)", fmt(report.get("rank_ic"), "{:.3f}"))
        excluded = report.get("excluded") or {}
        if excluded:
            st.markdown(f"**Excluded today ({len(excluded)}):**")
            st.dataframe(pd.DataFrame({"Ticker": list(excluded), "Reason": list(excluded.values())}), hide_index=True)


def render_live_scoreboard():
    live = dd.live_results(scores, hist)
    with st.expander("Live track record since launch"):
        if live.empty:
            st.info("No live results yet. They start appearing about two weeks after the first scored day, "
                    "once 10 trading days have passed.")
            return
        hit = dd.hit_summary(live)
        c1, c2, c3 = st.columns(3)
        c1.metric("Scored days with a known outcome", live["Date"].nunique())
        c2.metric("Right calls", "-" if not hit else str(hit["right"]) + " of " + str(hit["calls"]))
        c3.metric("Hit rate", "-" if not hit else fmt(hit["hit_rate"], "{:.0f}", "%"))
        table = dd.rating_table(live).reset_index().rename(columns={
            "n": "Calls", "avg_return": "Avg 10d return %", "avg_excess": "Avg vs market %"}).round(2)
        st.dataframe(table, hide_index=True)
        st.caption("A call is right when a Very Strong or Strong stock beat the market over the next 10 trading days, "
                   "or a Weak or Very Weak stock lagged it. A coin flip is right about 50% of the time.")


# ----------------------------------------------------------------------------- main page
def render_ranking():
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Stocks ranked", len(pred))
    c2.metric("Prices as of", str(asof.date()))
    c3.metric("Average score", fmt(pred["Score"].mean(), "{:.3f}"))
    c4.metric("Strong or Very Strong", int(pred["Rating"].isin(["Very Strong", "Strong"]).sum()))

    f1, f2 = st.columns([2, 3])
    query = f1.text_input("Search ticker or company", "")
    chosen = f2.multiselect("Rating", dd.RATINGS, default=dd.RATINGS)

    view = dd.ranking_view(pred, query, chosen or None)
    st.caption("Click a row to open the stock's details. Click a column header to sort.")

    config = {
        "Rank": st.column_config.NumberColumn("Rank", format="%d"),
        "Score": st.column_config.ProgressColumn(
            "Score", min_value=float(pred["Score"].min()), max_value=float(pred["Score"].max()), format="%.3f"),
        "Close": st.column_config.NumberColumn("Close", format="%.2f"),
        "1d %": st.column_config.NumberColumn("1d %", format="%.2f"),
        "5d %": st.column_config.NumberColumn("5d %", format="%.2f"),
        "20d %": st.column_config.NumberColumn("20d %", format="%.2f"),
        "RSI": st.column_config.NumberColumn("RSI", format="%.0f"),
    }
    event = st.dataframe(
        view, hide_index=True, height=620, column_config=config,
        on_select="rerun", selection_mode="single-row",
        key="ranking_" + str(st.session_state.get("nonce", 0)),
    )
    picked = event.selection.rows
    if picked:
        st.query_params["ticker"] = str(view.iloc[picked[0]]["Ticker"])
        st.rerun()

    render_live_scoreboard()
    render_model_card()


# ----------------------------------------------------------------------------- detail page
def price_figure(h):
    up, down = "#26a69a", "#ef5350"
    x = h["Date"].dt.strftime("%Y-%m-%d")
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                        row_heights=[0.50, 0.14, 0.18, 0.18])
    fig.add_trace(go.Candlestick(x=x, open=h["Open"], high=h["High"], low=h["Low"], close=h["Close"],
                                 name="Price", increasing_line_color=up, decreasing_line_color=down),
                  row=1, col=1)
    for col, name, color in (("MA_20", "MA 20", "#f6a800"), ("MA_50", "MA 50", "#2962ff"),
                             ("MA_200", "MA 200", "#9c27b0")):
        fig.add_trace(go.Scatter(x=x, y=h[col], name=name, mode="lines", line=dict(width=1.3, color=color)),
                      row=1, col=1)
    for col, name in (("BB_Upper", "Bollinger upper"), ("BB_Lower", "Bollinger lower")):
        fig.add_trace(go.Scatter(x=x, y=h[col], name=name, mode="lines",
                                 line=dict(width=1, color="#8a8f98", dash="dot")), row=1, col=1)

    fig.add_trace(go.Bar(x=x, y=h["Volume"], marker_color=np.where(h["Close"] >= h["Open"], up, down),
                         name="Volume", showlegend=False), row=2, col=1)

    fig.add_trace(go.Scatter(x=x, y=h["RSI_14"], name="RSI 14", mode="lines",
                             line=dict(width=1.5, color="#7e57c2")), row=3, col=1)
    fig.add_hline(y=70, line_dash="dot", line_color=down, row=3, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color=up, row=3, col=1)

    fig.add_trace(go.Bar(x=x, y=h["MACD_Hist_%"], marker_color=np.where(h["MACD_Hist_%"] >= 0, up, down),
                         name="MACD histogram", showlegend=False), row=4, col=1)
    fig.add_trace(go.Scatter(x=x, y=h["MACD_%"], name="MACD", mode="lines",
                             line=dict(width=1.3, color="#2962ff")), row=4, col=1)
    fig.add_trace(go.Scatter(x=x, y=h["MACD_Signal_%"], name="Signal", mode="lines",
                             line=dict(width=1.3, color="#f6a800")), row=4, col=1)

    fig.update_xaxes(type="category", nticks=8, rangeslider_visible=False)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_yaxes(title_text="RSI", range=[0, 100], row=3, col=1)
    fig.update_yaxes(title_text="MACD %", row=4, col=1)
    fig.update_layout(height=780, margin=dict(l=10, r=10, t=30, b=10), hovermode="x unified",
                      legend=dict(orientation="h", y=1.04, x=0))
    return fig


def rating_bar_figure(by):
    vals = by["avg_excess"]
    fig = go.Figure(go.Bar(
        x=list(by.index), y=vals,
        marker_color=["#26a69a" if (pd.notna(v) and v > 0) else "#ef5350" for v in vals],
        text=["no calls" if pd.isna(v) else f"{v:+.2f}%<br>n={n}" for v, n in zip(vals, by["n"])],
        textposition="outside", cliponaxis=False))
    fig.add_hline(y=0, line_color="#8a8f98")
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10), showlegend=False,
                      yaxis_title="Avg 10-day return vs market (%)")
    return fig


def rank_figure(s):
    x = s["Date"].dt.strftime("%Y-%m-%d")
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        row_heights=[0.5, 0.5], subplot_titles=("Rank (1 = best)", "Score"))
    fig.add_trace(go.Scatter(x=x, y=s["Rank"], mode="lines+markers", name="Rank",
                             line=dict(color="#2962ff")), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=s["Score"], mode="lines+markers", name="Score",
                             line=dict(color="#26a69a")), row=2, col=1)
    fig.update_yaxes(autorange="reversed", row=1, col=1)
    fig.update_xaxes(type="category")
    fig.update_layout(height=520, margin=dict(l=10, r=10, t=40, b=10), showlegend=False)
    return fig


def back_to_ranking():
    st.session_state["nonce"] = st.session_state.get("nonce", 0) + 1
    st.query_params.clear()
    st.rerun()


def hit_metrics(hit, market_hit):
    c1, c2, c3 = st.columns(3)
    c1.metric("Right calls", str(hit["right"]) + " of " + str(hit["calls"]))
    delta = None if not market_hit else fmt(hit["hit_rate"] - market_hit["hit_rate"], "{:+.0f}", " pts vs all stocks")
    c2.metric("Hit rate", fmt(hit["hit_rate"], "{:.0f}", "%"), delta)
    c3.metric("All stocks", "-" if not market_hit else fmt(market_hit["hit_rate"], "{:.0f}", "%"))


def render_track_record(ticker):
    st.caption("A call is **right** when a Very Strong or Strong stock beat the market over the next 10 trading days, "
               "or a Weak or Very Weak stock lagged it. Average ratings are not counted. "
               "A coin flip is right about 50% of the time. Wrong calls are shown too.")

    st.markdown("##### Live record: calls made before the outcome was known")
    live = dd.live_results(scores, hist)
    mine = live[live["Ticker"] == ticker] if not live.empty else live
    if mine.empty:
        first = scores["Date"].min() if len(scores) else None
        st.info("Live results appear about two weeks after the first scored day"
                + ("" if first is None else " (" + str(first.date()) + ")") + ", once 10 trading days have passed.")
    else:
        hit = dd.hit_summary(mine)
        if hit:
            hit_metrics(hit, dd.hit_summary(live))
        st.dataframe(dd.calls_table(mine), hide_index=True)

    st.markdown("##### Test period: days the model had not seen when it was trained")
    rec = dd.stock_track_record(backtest, ticker)
    if rec is None:
        st.info("The test-period record is created the next time the model retrains.")
        return
    if rec["hit"]:
        hit_metrics(rec["hit"], rec["market_hit"])
    st.plotly_chart(rating_bar_figure(rec["by_rating"]))
    st.warning("A rough guide, not proof. Each call looks 10 days ahead, so calls overlap and this stock has only about "
               + str(max(rec["days"] // 10, 1)) + " independent periods. The all-stocks numbers on the main page are more reliable.")
    st.dataframe(dd.calls_table(rec["rows"]), hide_index=True)


def render_detail(ticker):
    if st.button("← Back to ranking"):
        back_to_ranking()

    row = pred[pred["Ticker"] == ticker]
    if row.empty:
        st.warning(ticker + " is not in today's ranking (it may have been excluded).")
        return
    r = row.iloc[0]

    st.subheader(str(r["Company_Name"]) + "  ·  " + ticker)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Rank", "#" + str(int(r["Rank"])) + " of " + str(len(pred)))
    c2.metric("Rating", rating_label(r["Rating"]))
    c3.metric("Score", fmt(r["Score"], "{:.3f}"))
    c4.metric("Close", fmt(r["Close"]), fmt(r["Daily_Return_%"], "{:+.2f}", "%"))
    c5.metric("RSI (14)", fmt(r["RSI_14"], "{:.0f}"))

    for item in report.get("ratings") or []:
        if item.get("rating") == r["Rating"]:
            st.info(
                "In the test period, stocks rated **" + str(r["Rating"]) + "** returned on average "
                + fmt(item.get("avg_10d_return"), "{:+.2f}", "%") + " over 10 trading days ("
                + fmt(item.get("excess_vs_market"), "{:+.2f}", "%") + " vs the market), and "
                + fmt(item.get("pct_up"), "{:.0f}", "%") + " of them went up. This is a historical average, not a promise.")

    tab_chart, tab_ind, tab_rank, tab_proof = st.tabs(["Chart", "Indicators", "Rank history", "Track record"])

    with tab_chart:
        span = st.radio("Range", ["3M", "6M", "1Y"], index=1, horizontal=True)
        days = {"3M": 63, "6M": 126, "1Y": 252}[span]
        h = dd.ticker_history(hist, ticker, days)
        if h.empty:
            st.info("No price history for this stock.")
        else:
            st.plotly_chart(price_figure(h))

    with tab_ind:
        left, right = st.columns(2)
        stock_rows = [
            ("Return 1d %", fmt(r["Daily_Return_%"], "{:+.2f}"), ""),
            ("Return 5d %", fmt(r["Return_5d_%"], "{:+.2f}"), ""),
            ("Return 20d %", fmt(r["Return_20d_%"], "{:+.2f}"), ""),
            ("RSI (14)", fmt(r["RSI_14"], "{:.1f}"), "Above 70 is often called overbought, below 30 oversold"),
            ("MACD histogram %", fmt(r["MACD_Hist_%"], "{:+.3f}"), "Above 0 means upward momentum"),
            ("Price vs MA20 %", fmt(r["Price_to_MA20_%"], "{:+.2f}"), "Distance from the 20-day average"),
            ("Price vs MA50 %", fmt(r["Price_to_MA50_%"], "{:+.2f}"), ""),
            ("Price vs MA200 %", fmt(r["Price_to_MA200_%"], "{:+.2f}"), ""),
            ("Bollinger position", fmt(r["BB_Position"], "{:.2f}"), "0 = lower band, 1 = upper band"),
            ("Relative volume", fmt(r["Relative_Volume"], "{:.2f}"), "1 = normal volume"),
            ("Volatility 30d", fmt(r["Volatility_30d"], "{:.2f}"), ""),
        ]
        left.markdown("**This stock**")
        left.dataframe(pd.DataFrame(stock_rows, columns=["Indicator", "Value", "Meaning"]), hide_index=True)
        market_rows = [
            ("News sentiment", fmt(r["Sentiment_Score"], "{:.3f}")),
            ("News volume", fmt(r["News_Volume"], "{:,.0f}")),
            ("USD/EGP", fmt(r["USD_EGP"], "{:.2f}")),
            ("Brent oil", fmt(r["Brent_Oil"], "{:.2f}")),
            ("Market above MA20 %", fmt(r["Market_Above_MA20_%"], "{:.1f}")),
        ]
        right.markdown("**Market context (same for all stocks)**")
        right.dataframe(pd.DataFrame(market_rows, columns=["Item", "Value"]), hide_index=True)

    with tab_rank:
        s = dd.ticker_rank_history(scores, ticker)
        if len(s) < 2:
            st.info("Rank history builds up one day at a time. Days collected so far: " + str(len(s)) + ".")
        else:
            st.plotly_chart(rank_figure(s))

    with tab_proof:
        render_track_record(ticker)


# ----------------------------------------------------------------------------- run
render_header()
selected = st.query_params.get("ticker")
if selected:
    render_detail(str(selected))
else:
    render_ranking()

st.divider()
st.caption("Model output for education only, not investment advice. Ratings are relative to the other stocks on the same day.")
