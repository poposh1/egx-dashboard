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
expl = data["explanations"]
news_imputed = data.get("news_imputed")      # days whose news numbers were copied from an earlier day (or None)
asof = pd.Timestamp(report.get("asof") or pred["Date"].max())

upside = dd.upside_table(pred, backtest)      # expected 10-day change per stock (from the test period)
calib = dd.score_calibration(backtest)        # what each ranking band actually returned in the test period
drivers = dd.main_driver_map(expl, pred)      # the one factor that moved each stock's score most
cutoffs = dd.rating_cutoffs(pred)             # score range behind each rating today
backdrop = dd.market_backdrop(expl)           # today's market-wide push on every score


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
    for msg in dd.build_warnings(status, report, asof, news_imputed):
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


# ----------------------------------------------------------------------------- guide
def render_guide():
    """شرح كل رقم في الداشبورد في مكان واحد (يتفتح من غير ما تسيب الصفحة)."""
    with st.expander("📖 Guide: what every number on this dashboard means", expanded=False):
        st.markdown(
            "**What this page does.** Every trading day the model scores the stocks of the Egyptian exchange and "
            "ranks them for the next 10 trading days. It uses price, volume and market indicators, plus two small "
            "daily market-wide numbers about news (how many articles and their average tone) - it never reads the "
            "news text, and it uses no fundamentals. It is a course project, **not investment advice**."
        )
        st.markdown("##### The ranking table")
        st.markdown(
            "- **Rank** - today's position. 1 = the highest score of the day.\n"
            "- **Score** - the model's output, between 0 and 1: how likely this stock is a *strong riser* "
            "(up 3.5% or more within 10 trading days) rather than a *strong faller* (down 3% or more). "
            "It was learned from 2020-2026 price history and is only used to rank stocks against each other.\n"
            "- **Rating** - where the score falls in today's list: top 20% = Very Strong, next 20% = Strong, then "
            "Average, Weak, bottom 20% = Very Weak. It is **relative**: the same number of stocks carry each label "
            "every day, even in a falling market. \"Very Strong\" is not a broker's \"Strong Buy\".\n"
            "- **Expected 10d %** - what stocks standing in the same band of the ranking actually returned over the "
            "next 10 trading days during the test period. It is the average for that band, not a forecast for this "
            "one stock.\n"
            "- **Typical range** - the middle half of those outcomes (25th to 75th percentile). Wide on purpose: "
            "10-day moves are noisy.\n"
            "- **Main reason** - the single factor that moved this stock's score most, compared with the other stocks "
            "today. ↑ pushed its score up, ↓ pulled it down.\n"
            "- **1d % / 5d % / 20d %** - the stock's own price change over 1, 5 and 20 trading days.\n"
            "- **RSI** - 14-day Relative Strength Index. Above 70 is usually called overbought, below 30 oversold.\n"
            "- **Close** - the last closing price in EGP."
        )
        st.markdown("##### Why the rating is what it is today")
        if not cutoffs.empty:
            cut = pd.DataFrame({
                "Rating": [rating_label(r) for r in cutoffs.index],
                "Stocks": cutoffs["Stocks"].astype(int).values,
                "Lowest score in the band": cutoffs["Min score"].round(4).values,
                "Highest score in the band": cutoffs["Max score"].round(4).values,
            })
            st.dataframe(cut, hide_index=True)
        st.caption("The ratings are cut-offs on today's scores, not fixed levels: the same score can be Very Strong "
                   "on a weak day and Average on a strong one.")

        st.markdown("##### Where the expected 10-day % comes from")
        if calib.empty:
            st.info("The test-period file (backtest.csv) is not there yet, so the expected numbers cannot be shown. "
                    "They appear after the next model retrain.")
        else:
            table = pd.DataFrame({
                "Where it stood in the ranking": calib["Position"].values,
                "Cases": calib["Similar cases"].astype(int).values,
                "Avg 10d return %": calib["Expected_10d_%"].round(2).values,
                "% that rose": calib["P_up_%"].round(0).values,
                "Typical low %": calib["Range_Low_%"].round(1).values,
                "Typical high %": calib["Range_High_%"].round(1).values,
            })
            st.dataframe(table, hide_index=True)
            st.caption("Test period only: for every day in the test period, stocks were put in 10 bands by their "
                       "score, then the real 10-day return was measured. This is history, not a promise.")

        st.markdown("##### How good is the model, honestly")
        st.markdown(
            f"- Rank correlation with what actually happened (IC): **{fmt(report.get('rank_ic'), '{:.3f}')}** "
            "(0 = no skill, 1 = perfect).\n"
            f"- Strong-move detection (AUC): **{fmt(report.get('auc_strong_moves'), '{:.4f}')}** (0.5 = a coin flip).\n"
            f"- In the test period the top 10 stocks each day returned **{fmt(report.get('top10_daily_avg_return'), '{:+.2f}', '%')}** "
            f"on average over the next 10 days, the bottom 10 **{fmt(report.get('bottom10_daily_avg_return'), '{:+.2f}', '%')}**.\n"
            f"- The whole market averaged **{fmt(report.get('market_avg_10d_return'), '{:+.2f}', '%')}** over the same horizon, "
            "so most of any 10-day return is the market, not the ranking."
        )
        st.caption("Translation: the model separates the strongest stocks from the weakest, but the edge is small and "
                   "the middle of the list is close to a tie.")

        st.markdown("##### The detail page (click any row)")
        st.markdown(
            "- **Why this score** - the factors behind this stock's score and rank: what the model looked at, "
            "this stock's value, the market median, and which way each factor pushed.\n"
            "- **Chart** - candlesticks with the 20/50/200-day averages, Bollinger bands, volume, RSI and MACD.\n"
            "- **Indicators** - every indicator the model uses for this stock, with a one-line meaning, plus the "
            "market context that is the same for all stocks.\n"
            "- **Rank history** - how the stock's rank and score moved day by day since the dashboard started.\n"
            "- **Track record** - how this stock's past calls turned out, live and in the test period."
        )

        st.markdown("##### Warnings and exclusions")
        st.markdown(
            "- A warning at the top means the data is old, the latest news numbers were copied from an earlier day (the news feed failed), a stock has no recent prices, or "
            "today's average score is outside the range the model was tested on. Rankings are less reliable then.\n"
            "- Stocks are excluded when they have not traded in the last 10 days, when their price history is broken, "
            "or when the indicators cannot be computed. The list is at the bottom of this guide."
        )
        excluded = report.get("excluded") or {}
        if excluded:
            st.dataframe(pd.DataFrame({"Ticker": list(excluded), "Why": list(excluded.values())}), hide_index=True)


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

    view = dd.ranking_view(pred, query, chosen or None, upside, drivers)
    st.caption("Click a row to open the stock's details. Click a column header to sort. "
               "Every column is explained in the guide above.")

    config = {
        "Rank": st.column_config.NumberColumn("Rank", format="%d", help="Today's position; 1 = the highest score."),
        "Score": st.column_config.NumberColumn(
            "Score", format="%.3f",
            help="Model output between 0 and 1: chance the stock is a strong riser (up 3.5% or more in 10 trading "
                 "days) rather than a strong faller. Used only to rank stocks against each other."),
        "Rating": st.column_config.TextColumn(
            "Rating", help="Position in today's list: top 20% Very Strong ... bottom 20% Very Weak. Relative, "
                           "not a broker's Strong Buy."),
        "Expected 10d %": st.column_config.NumberColumn(
            "Expected 10d %", format="%+.2f%%",
            help="Average 10-day return of stocks in the same ranking band during the test period. A band average, "
                 "not a per-stock forecast."),
        "Typical range": st.column_config.TextColumn(
            "Typical range", help="Middle half (25th-75th percentile) of those past outcomes."),
        "Close": st.column_config.NumberColumn("Close", format="%.2f", help="Last close in EGP."),
        "1d %": st.column_config.NumberColumn("1d %", format="%.2f"),
        "5d %": st.column_config.NumberColumn("5d %", format="%.2f"),
        "20d %": st.column_config.NumberColumn("20d %", format="%.2f"),
        "RSI": st.column_config.NumberColumn("RSI", format="%.0f",
                                             help="14-day RSI: above 70 usually called overbought, below 30 oversold."),
        "Main reason": st.column_config.TextColumn(
            "Main reason", width="medium",
            help="The factor that moved this stock's score most, compared with the other stocks today. "
                 "↑ pushed the score up, ↓ pulled it down."),
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


def driver_figure(d):
    """Bar chart: how much each factor pushed this stock's score, compared with the average stock today."""
    d = d.sort_values("Effect")
    colors = ["#26a69a" if v > 0 else "#ef5350" for v in d["Effect"]]
    fig = go.Figure(go.Bar(
        x=d["Effect"], y=d["Factor"], orientation="h", marker_color=colors,
        text=[f"{v:+.3f}" for v in d["Effect"]], textposition="outside", cliponaxis=False,
        customdata=np.stack([d["This stock"], d["Market median"], d["Position in market"]], axis=-1),
        hovertemplate="%{y}<br>this stock: %{customdata[0]} (market median %{customdata[1]})"
                      "<br>position in market: %{customdata[2]}<extra></extra>"))
    fig.add_vline(x=0, line_color="#8a8f98")
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10), showlegend=False,
                      xaxis_title="Effect on the score (model units, vs the average stock today)")
    return fig


def render_why(ticker, r):
    """ليه السهم ده واقف في المكان ده: الترتيب والدرجة والعوامل، واللي فترة الاختبار بتقوله."""
    n = len(pred)
    rank = int(r["Rank"])
    st.markdown(f"##### Why {ticker} is ranked #{rank} of {n}")

    above = pred[pred["Rank"] < rank]["Score"]
    below = pred[pred["Rank"] > rank]["Score"]
    if rank == 1:
        where = "It is the highest score of the day."
    elif rank == n:
        where = "It is the lowest score of the day."
    else:
        where = (f"The stock just above it scores {fmt(above.min(), '{:.3f}')}, the one just below "
                 f"{fmt(below.max(), '{:.3f}')}.")
    bullets = [
        f"**Score {fmt(r['Score'], '{:.3f}')}** puts it at rank **#{rank} of {n}**. " + where,
        f"**What the score means:** the model gives it a {r['Score']:.0%} chance of being a strong riser "
        "(up 3.5% or more in 10 trading days) rather than a strong faller (down 3% or more), judged from price and "
        "volume patterns learned on 2020-2026 data.",
    ]
    if not cutoffs.empty and r["Rating"] in cutoffs.index:
        c = cutoffs.loc[r["Rating"]]
        band = {"Very Strong": "top 20% of today's scores", "Strong": "next 20% down from the top",
                "Average": "the middle 20%", "Weak": "the 20% below the middle",
                "Very Weak": "the bottom 20%"}.get(r["Rating"], "a slice of today's ranking")
        bullets.insert(1, f"**Rating {r['Rating']}** = {band}: today that means a score from "
                          f"{c['Min score']:.3f} to {c['Max score']:.3f} ({int(c['Stocks'])} stocks). Ratings are "
                          "relative to today's list, not fixed levels, and not a broker's Strong Buy.")
    else:
        bullets.insert(1, f"**Rating {r['Rating']}** comes from its place in today's list.")
    mine = upside[upside["Ticker"] == ticker]
    if not mine.empty:
        m = mine.iloc[0]
        bullets.append(f"**What to expect:** stocks that stood in the **{m['Position']}** during the test period "
                       f"returned **{m['Expected_10d_%']:+.2f}%** on average over the next 10 trading days "
                       f"({int(m['Similar cases']):,} cases), rose {m['P_up_%']:.0f}% of the time, and the middle half "
                       f"landed between {m['Range_Low_%']:+.1f}% and {m['Range_High_%']:+.1f}%. That is a band average, "
                       "not a promise for this stock.")
    for b in bullets:
        st.markdown("- " + b)

    st.markdown("##### What moved the score")
    st.caption("Each factor's push on this stock's score, compared with the average stock today. "
               "Green pushed the score up, red pulled it down. Factors that are identical for every stock "
               "(the market context) are left out because they do not change the ranking.")
    d = dd.driver_table(expl, ticker, top=10)
    if d.empty:
        st.info("No explanation file for today yet. It is written by scripts/explain.py during the daily update.")
        return
    st.plotly_chart(driver_figure(dd.driver_frame(expl, ticker, top=10)))

    bullets = dd.why_bullets(expl, ticker, n=4)
    if bullets:
        st.markdown("**In plain words:**")
        for b in bullets:
            st.markdown("- " + b)

    frame = dd.driver_frame(expl, ticker, top=10)
    st.dataframe(
        frame, hide_index=True,
        column_config={
            "Effect": st.column_config.NumberColumn("Effect", format="%+.3f",
                                                    help="Model units. Positive = this factor pushed the score up."),
            "Position in market": st.column_config.TextColumn(
                "Position in market", help="Where this stock's value sits among today's stocks (0% = lowest, 100% = highest)."),
        })
    if backdrop:
        st.info(f"Today's market backdrop ({', '.join(backdrop['features'])}) moves every stock's score by about "
                f"{backdrop['effect']:+.2f} model units. It is the same for all stocks, so it does not change the ranking.")
    st.caption("The model never reads news or company results. These factors are price, volume and market "
               "indicators only, so a factor can be right about the past and wrong about next week.")


def render_detail(ticker):
    if st.button("← Back to ranking"):
        back_to_ranking()

    row = pred[pred["Ticker"] == ticker]
    if row.empty:
        st.warning(ticker + " is not in today's ranking (it may have been excluded).")
        return
    r = row.iloc[0]

    st.subheader(str(r["Company_Name"]) + "  ·  " + ticker)
    mine = upside[upside["Ticker"] == ticker]
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Rank", "#" + str(int(r["Rank"])) + " of " + str(len(pred)))
    c2.metric("Rating", rating_label(r["Rating"]))
    c3.metric("Score", fmt(r["Score"], "{:.3f}"))
    if mine.empty:
        c4.metric("Expected 10d %", "-")
    else:
        m = mine.iloc[0]
        c4.metric("Expected 10d %", fmt(m["Expected_10d_%"], "{:+.1f}", "%"),
                  f"range {m['Range_Low_%']:+.1f}% to {m['Range_High_%']:+.1f}%", delta_color="off")
    c5.metric("Close", fmt(r["Close"]), fmt(r["Daily_Return_%"], "{:+.2f}", "%"))
    c6.metric("RSI (14)", fmt(r["RSI_14"], "{:.0f}"))

    for item in report.get("ratings") or []:
        if item.get("rating") == r["Rating"]:
            st.info(
                "In the test period, stocks rated **" + str(r["Rating"]) + "** that day (that is, the same fifth "
                "of the list) returned on average "
                + fmt(item.get("avg_10d_return"), "{:+.2f}", "%") + " over 10 trading days ("
                + fmt(item.get("excess_vs_market"), "{:+.2f}", "%") + " vs the market), and "
                + fmt(item.get("pct_up"), "{:.0f}", "%") + " of them went up. This is a historical average, not a promise.")

    tab_why, tab_chart, tab_ind, tab_rank, tab_proof = st.tabs(
        ["Why this score", "Chart", "Indicators", "Rank history", "Track record"])

    with tab_why:
        render_why(ticker, r)

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
        carried = dd.news_is_carried(asof, news_imputed)
        copied = "  ⚠ copied from an earlier day" if carried else ""
        market_rows = [
            ("News sentiment", fmt(r["Sentiment_Score"], "{:.3f}") + copied),
            ("News volume", fmt(r["News_Volume"], "{:,.0f}") + copied),
            ("USD/EGP (Yahoo market rate)", fmt(r["USD_EGP"], "{:.2f}")),
            ("Brent oil", fmt(r["Brent_Oil"], "{:.2f}")),
            ("Market above MA20 %", fmt(r["Market_Above_MA20_%"], "{:.1f}")),
        ]
        right.markdown("**Market context (same for all stocks)**")
        right.dataframe(pd.DataFrame(market_rows, columns=["Item", "Value"]), hide_index=True)
        if carried:
            right.caption("⚠ The news feed failed for this day, so the news numbers are copied from an earlier day. "
                          "They are retried automatically on the next update.")
        right.caption(f"USD/EGP is the Yahoo Finance market rate for {asof.date()}, not a bank's buy or sell price. "
                      "Banks and the Central Bank usually quote within about 0.5% of it, and the rate may have moved since.")

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
render_guide()
selected = st.query_params.get("ticker")
if selected:
    render_detail(str(selected))
else:
    render_ranking()

st.divider()
st.caption("Model output for education only, not investment advice. Ratings are relative to the other stocks on the same day.")
