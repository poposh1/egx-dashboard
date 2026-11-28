"""Data helpers for the dashboard. No Streamlit imports here, so this file can be tested on its own."""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(os.environ.get("EGX_DATA_DIR") or Path(__file__).resolve().parent / "data")

PRED_CSV = DATA_DIR / "predictions.csv"
HISTORY_CSV = DATA_DIR / "dashboard_history.csv"
SCORES_CSV = DATA_DIR / "scores_history.csv"
REPORT_JSON = DATA_DIR / "model_report.json"
STATUS_JSON = DATA_DIR / "update_status.json"
BACKTEST_CSV = DATA_DIR / "backtest.csv"
EXPLAIN_CSV = DATA_DIR / "explanations.csv"     # per-stock feature contributions (scripts/explain.py)
IMPUTED_JSON = DATA_DIR / "news_imputed_dates.json"   # days whose news numbers were copied from an earlier day

RATINGS = ["Very Strong", "Strong", "Average", "Weak", "Very Weak"]
RATING_EMOJI = {"Very Strong": "🟢", "Strong": "🔵", "Average": "⚪", "Weak": "🟠", "Very Weak": "🔴"}


def files_signature():
    """Modification times of the data files. Used as the cache key: when a file changes, the data reloads."""
    return tuple(p.stat().st_mtime_ns if p.exists() else 0
                 for p in (PRED_CSV, HISTORY_CSV, SCORES_CSV, REPORT_JSON, STATUS_JSON, BACKTEST_CSV,
                           EXPLAIN_CSV, IMPUTED_JSON))


def _read_json(path):
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def load_news_imputed():
    """Dates (Timestamps) whose News_Volume / Sentiment_Score were copied from an earlier day.
    Returns None when the file does not exist (older data folders), so callers can fall back to update_status.json."""
    if not IMPUTED_JSON.exists():
        return None
    try:
        raw = json.loads(IMPUTED_JSON.read_text(encoding="utf-8")).get("dates", [])
        return sorted(pd.to_datetime(raw).normalize())
    except Exception:
        return None


def news_is_carried(asof, imputed):
    """True when the news numbers shown for `asof` were copied from an earlier day."""
    if not imputed:
        return False
    return pd.Timestamp(asof).normalize() in set(imputed)


def load_all():
    missing = [p.name for p in (PRED_CSV, HISTORY_CSV) if not p.exists()]
    if missing:
        raise FileNotFoundError(", ".join(missing))
    pred = pd.read_csv(PRED_CSV, parse_dates=["Date"])
    hist = pd.read_csv(HISTORY_CSV, parse_dates=["Date"])
    if SCORES_CSV.exists():
        scores = pd.read_csv(SCORES_CSV, parse_dates=["Date"])
    else:
        scores = pd.DataFrame(columns=["Date", "Ticker", "Score", "Rank", "Rating"])
    backtest = pd.read_csv(BACKTEST_CSV, parse_dates=["Date"]) if BACKTEST_CSV.exists() else pd.DataFrame()
    explain = pd.read_csv(EXPLAIN_CSV) if EXPLAIN_CSV.exists() else pd.DataFrame(
        columns=["Date", "Ticker", "Feature", "Contribution", "Value", "Market_Median", "Value_Pct_Rank"])
    return {
        "predictions": pred.sort_values("Rank").reset_index(drop=True),
        "history": hist,
        "scores": scores,
        "report": _read_json(REPORT_JSON),
        "status": _read_json(STATUS_JSON),
        "news_imputed": load_news_imputed(),
        "backtest": backtest,
        "explanations": explain,
    }


def ranking_view(pred, query="", ratings=None, upside=None, drivers=None):
    """The table shown on the main page: sorted by Rank (best first), optionally filtered.

    `upside`  : per-ticker expected 10-day change (see upside_table); adds the "Expected 10d %" column.
    `drivers` : per-ticker short text for the strongest stock-specific driver (see main_driver_map).
    """
    df = pred.sort_values("Rank")
    if ratings:
        df = df[df["Rating"].isin(ratings)]
    q = (query or "").strip().lower()
    if q:
        in_ticker = df["Ticker"].astype(str).str.lower().str.contains(q, regex=False)
        in_name = df["Company_Name"].fillna("").astype(str).str.lower().str.contains(q, regex=False)
        df = df[in_ticker | in_name]
    out = pd.DataFrame({
        "Rank": df["Rank"],
        "Ticker": df["Ticker"],
        "Company": df["Company_Name"],
        "Rating": [RATING_EMOJI.get(r, "") + " " + str(r) for r in df["Rating"]],
        "Score": df["Score"],
    })
    if upside is not None and not upside.empty:
        up = upside.set_index("Ticker")
        out["Expected 10d %"] = df["Ticker"].map(up["Expected_10d_%"]).values
        out["Typical range"] = df["Ticker"].map(
            up.apply(lambda r: f"{r['Range_Low_%']:+.1f}% to {r['Range_High_%']:+.1f}%", axis=1)).values
    out["Close"] = df["Close"].values
    out["1d %"] = df["Daily_Return_%"].values
    out["5d %"] = df["Return_5d_%"].values
    out["20d %"] = df["Return_20d_%"].values
    out["RSI"] = df["RSI_14"].values
    if drivers:
        out["Main reason"] = df["Ticker"].map(drivers).values
    return out.reset_index(drop=True)


def ticker_history(hist, ticker, days):
    h = hist[hist["Ticker"] == ticker].sort_values("Date")
    return h.tail(days).reset_index(drop=True)


def ticker_rank_history(scores, ticker):
    s = scores[scores["Ticker"] == ticker].sort_values("Date")
    return s.reset_index(drop=True)


def data_age_days(asof):
    today = pd.Timestamp.now(tz="Africa/Cairo").tz_localize(None).normalize()
    return int((today - pd.Timestamp(asof).normalize()).days)


def build_warnings(status, report, asof, news_imputed=None):
    """Plain-language warnings shown at the top of the dashboard."""
    msgs = []
    age = data_age_days(asof)
    if age >= 5:
        msgs.append(f"The latest data is {age} days old. The daily update may have failed - check the GitHub Actions tab.")
    if news_imputed is not None:
        # We know exactly which days were copied: warn only if the latest day is one of them.
        if news_is_carried(asof, news_imputed):
            msgs.append("The news numbers for the latest day are copied from an earlier day, because the news feed "
                        "failed. The next update retries them. News factors carry very little weight in the ranking.")
    elif status and status.get("news_ok") is False:       # older data folder without news_imputed_dates.json
        msgs.append("News data could not be fetched in the last update. Sentiment values for the newest days are "
                    "carried forward from the last available day, so rankings may be slightly less accurate.")
    stale = (status or {}).get("stale_tickers") or {}
    if stale:
        msgs.append(f"{len(stale)} stock(s) have no recent price data: {', '.join(list(stale)[:8])}.")
    if report.get("market_score_above_test_range"):
        msgs.append("Today's average score is above the range the model saw during testing. "
                    "Treat the scores with extra caution.")
    return msgs


# ----------------------------------------------------------------------------- track record
TOP = ["Very Strong", "Strong"]
BOTTOM = ["Weak", "Very Weak"]
HORIZON = 10


def call_result(df):
    """'right' when a top-40% stock beat the market or a bottom-40% stock lagged it, 'wrong' otherwise, '-' for Average."""
    top, bottom = df["Rating"].isin(TOP), df["Rating"].isin(BOTTOM)
    right = (top & (df["Excess"] > 0)) | (bottom & (df["Excess"] < 0))
    return np.select([right, top | bottom], ["right", "wrong"], default="-")


def hit_summary(df):
    if df is None or df.empty:
        return None
    res = call_result(df)
    calls = int((res != "-").sum())
    if calls == 0:
        return None
    right = int((res == "right").sum())
    return {"calls": calls, "right": right, "hit_rate": right / calls * 100}


def rating_table(df):
    g = df.groupby("Rating")
    out = pd.DataFrame({"n": g.size(), "avg_return": g["Future_Return_10d_%"].mean(),
                        "avg_excess": g["Excess"].mean()}).reindex(RATINGS).rename_axis("Rating")
    out["n"] = out["n"].fillna(0).astype(int)
    return out


def stock_track_record(bt, ticker):
    """Test-period record (backtest.csv) for one stock."""
    if bt is None or bt.empty:
        return None
    s = bt[bt["Ticker"] == ticker].sort_values("Date")
    if s.empty:
        return None
    return {"rows": s, "hit": hit_summary(s), "market_hit": hit_summary(bt),
            "by_rating": rating_table(s), "days": int(len(s))}


def live_results(scores, hist, horizon=HORIZON):
    """Past daily scores joined with what really happened over the next `horizon` trading days."""
    if scores is None or scores.empty:
        return pd.DataFrame()
    px = hist[["Date", "Ticker", "Close"]].sort_values(["Ticker", "Date"]).copy()
    px["Future_Return_10d_%"] = (px.groupby("Ticker")["Close"].shift(-horizon) / px["Close"] - 1) * 100
    m = scores.merge(px[["Date", "Ticker", "Future_Return_10d_%"]], on=["Date", "Ticker"], how="left")
    m = m.dropna(subset=["Future_Return_10d_%"]).copy()
    if m.empty:
        return m
    m["Excess"] = m["Future_Return_10d_%"] - m.groupby("Date")["Future_Return_10d_%"].transform("mean")
    return m


def calls_table(df, n=30):
    t = df.sort_values("Date", ascending=False).head(n)
    return pd.DataFrame({
        "Date": t["Date"].dt.date,
        "Rating": [RATING_EMOJI.get(r, "") + " " + str(r) for r in t["Rating"]],
        "Score": t["Score"].round(3),
        "Actual 10d %": t["Future_Return_10d_%"].round(2),
        "vs market %": t["Excess"].round(2),
        "Result": [{"right": "\u2705 right", "wrong": "\u274c wrong"}.get(x, "-") for x in call_result(t)],
    }).reset_index(drop=True)


# ----------------------------------------------------------------------------- expected 10-day change
# الترتيب النهارده بيقول السهم أحسن من غيره، لكنه مش بيقول "متوقع يزيد كام".
# الحل: ناخد فترة الاختبار (backtest.csv: توقع + اللي حصل فعلًا بعد 10 أيام)، ونشوف كل سهم
# كان واقف فين في ترتيب يومه، ونحسب متوسط اللي حصل لأسهم كانت في نفس المكان.
# الناتج: "متوقع +2.5% في 10 أيام، وطلع في 60% من الحالات المشابهة".
CALIB_BUCKETS = 10


def _bucket_label(bucket):
    if bucket == 0:
        return "bottom 10% of the ranking"
    if bucket == CALIB_BUCKETS - 1:
        return "top 10% of the ranking"
    return f"{bucket * 10}-{(bucket + 1) * 10}% of the ranking"


def score_calibration(backtest):
    """Historical outcome per ranking position (from the test period). Empty frame when no backtest."""
    if backtest is None or backtest.empty:
        return pd.DataFrame()
    bt = backtest.dropna(subset=["Score", "Future_Return_10d_%"]).copy()
    if bt.empty:
        return pd.DataFrame()
    bt["Pct"] = bt.groupby("Date")["Score"].rank(pct=True)
    bt["Bucket"] = np.clip(np.ceil(bt["Pct"] * CALIB_BUCKETS) - 1, 0, CALIB_BUCKETS - 1).astype(int)
    g = bt.groupby("Bucket")
    out = pd.DataFrame({
        "Percentile_Bucket": g.size().index,
        "Similar cases": g.size().values,
        "Expected_10d_%": g["Future_Return_10d_%"].mean().values,
        "P_up_%": g["Future_Return_10d_%"].apply(lambda s: (s > 0).mean() * 100).values,
        "Range_Low_%": g["Future_Return_10d_%"].quantile(0.25).values,
        "Range_High_%": g["Future_Return_10d_%"].quantile(0.75).values,
    })
    out["Position"] = [_bucket_label(b) for b in out["Percentile_Bucket"]]
    return out


def upside_table(pred, backtest):
    """Expected 10-day change per stock: its rank today, matched to what that rank position did in testing."""
    calib = score_calibration(backtest)
    if calib.empty or pred is None or pred.empty:
        return pd.DataFrame()
    df = pred[["Ticker", "Rank", "Rating", "Score"]].copy()
    df["Pct"] = df["Score"].rank(pct=True)
    df["Bucket"] = np.clip(np.ceil(df["Pct"] * CALIB_BUCKETS) - 1, 0, CALIB_BUCKETS - 1).astype(int)
    out = df.merge(calib, left_on="Bucket", right_on="Percentile_Bucket", how="left")
    return out[["Ticker", "Rank", "Rating", "Bucket", "Position", "Similar cases", "Expected_10d_%",
                "P_up_%", "Range_Low_%", "Range_High_%"]].sort_values("Rank").reset_index(drop=True)


# ----------------------------------------------------------------------------- why this stock
# كل feature: اسم مفهوم + شكل الرقم + معنى القيمة الواطية/العالية (اللي بتتقال في الشرح).
FEATURE_INFO = {
    "Price_to_MA200_%": {"label": "Price vs 200-day average", "fmt": "{:+.1f}%",
                         "low": "well below its long-term average, and the model reads that gap as room to bounce",
                         "high": "above its long-term average, so less of a beaten-down bounce setup"},
    "Price_to_MA50_%": {"label": "Price vs 50-day average", "fmt": "{:+.1f}%",
                        "low": "below its 3-month average", "high": "above its 3-month average"},
    "Price_to_MA20_%": {"label": "Price vs 20-day average", "fmt": "{:+.1f}%",
                        "low": "below its 1-month average", "high": "above its 1-month average"},
    "MA50_to_MA200_%": {"label": "50-day average vs 200-day average", "fmt": "{:+.1f}%",
                        "low": "the 50-day average sits below the 200-day average (a downtrend, which often reversed in the training years)",
                        "high": "the 50-day average sits above the 200-day average (an uptrend)"},
    "MA20_to_MA50_%": {"label": "20-day average vs 50-day average", "fmt": "{:+.1f}%",
                       "low": "the 20-day average is below the 50-day average", "high": "the 20-day average is above the 50-day average"},
    "Return_5d_%": {"label": "5-day return", "fmt": "{:+.1f}%", "low": "a weak last week", "high": "a strong last week"},
    "Return_10d_%": {"label": "10-day return", "fmt": "{:+.1f}%", "low": "a weak last two weeks", "high": "a strong last two weeks"},
    "Return_20d_%": {"label": "20-day return", "fmt": "{:+.1f}%", "low": "a weak last month", "high": "a strong last month"},
    "RSI_14": {"label": "RSI (14)", "fmt": "{:.1f}",
               "low": "oversold territory (below 30 is usually called oversold)", "high": "overbought territory (above 70)"},
    "Stoch_14": {"label": "Stochastic (14)", "fmt": "{:.1f}",
                 "low": "near the bottom of its recent range", "high": "near the top of its recent range"},
    "MACD_Hist_%": {"label": "MACD histogram", "fmt": "{:+.3f}%",
                    "low": "momentum is negative", "high": "momentum is positive"},
    "MACD_%": {"label": "MACD", "fmt": "{:+.2f}%", "low": "below its signal line", "high": "above its signal line"},
    "BB_Position": {"label": "Bollinger position", "fmt": "{:.2f}",
                    "low": "near the lower Bollinger band", "high": "near the upper Bollinger band"},
    "BB_Width_%": {"label": "Bollinger width", "fmt": "{:.1f}%",
                   "low": "a narrow trading range lately", "high": "a wide trading range lately"},
    "Relative_Volume": {"label": "Volume vs its 20-day avg", "fmt": "{:.2f}x",
                        "low": "trading on lighter volume than usual", "high": "trading on heavier volume than usual"},
    "Volatility_30d": {"label": "30-day volatility", "fmt": "{:.2f}%",
                       "low": "calmer than most stocks", "high": "more volatile than most stocks"},
    "Stock_Beta_Proxy": {"label": "Beta vs the market", "fmt": "{:.2f}",
                         "low": "moving less than the market", "high": "moving more than the market"},
    "Alpha_5d": {"label": "5-day return vs the market", "fmt": "{:+.1f}%",
                 "low": "it lagged the market this week", "high": "it beat the market this week"},
    "Alpha_MA5": {"label": "Average daily lead/lag vs the market", "fmt": "{:+.2f}%",
                  "low": "it has been lagging the market", "high": "it has been beating the market"},
    "Vol_Adjusted_Return_5d": {"label": "5-day return per unit of volatility", "fmt": "{:+.2f}",
                               "low": "its recent move is small for its usual swings", "high": "a large move for its usual swings"},
    "Stock_USD_Effect": {"label": "Return x USD/EGP move", "fmt": "{:+.2f}",
                         "low": "the pound move has not helped it", "high": "the pound move has helped it"},
    "Stock_News_Response": {"label": "Return x news sentiment", "fmt": "{:+.3f}",
                            "low": "it moved against the news tone", "high": "it moved with the news tone"},
    "Volume_Driven_Sentiment": {"label": "Volume x news sentiment", "fmt": "{:+.3f}",
                                "low": "news tone and volume point different ways", "high": "news tone and volume point the same way"},
    "Stock_x_Market_Trend": {"label": "Its trend rank x market trend", "fmt": "{:+.2f}",
                             "low": "weak trend in a weak market", "high": "strong trend in a strong market"},
    "Market_Above_MA20_%": {"label": "Market index vs its 20-day avg", "fmt": "{:+.2f}%",
                            "low": "the whole market is below its 1-month average", "high": "the whole market is above its 1-month average"},
    "Market_Above_MA50_%": {"label": "Market index vs its 50-day avg", "fmt": "{:+.2f}%",
                            "low": "the whole market is below its 3-month average", "high": "the whole market is above its 3-month average"},
}
_RANK_FEATURES = {
    "Volatility_30d_Rank": "Volatility rank in today's market",
    "MACD_Hist_%_Rank": "Momentum rank in today's market",
    "BB_Position_Rank": "Bollinger position rank in today's market",
    "Stoch_14_Rank": "Stochastic rank in today's market",
    "RSI_14_Rank": "RSI rank in today's market",
    "Return_20d_%_Rank": "1-month return rank in today's market",
    "Relative_Volume_Rank": "Volume rank in today's market",
}
for _f, _label in _RANK_FEATURES.items():
    FEATURE_INFO[_f] = {"label": _label, "fmt": "{:.0%}",
                        "low": "lower than most stocks today", "high": "higher than most stocks today"}


# جمل قصيرة لعمود "Main reason" في جدول الترتيب (السهم ده مختلف في إيه عن باقي السوق).
SHORT_PHRASES = {
    "Price_to_MA200_%": ("price {v:.0f}% below its 200-day avg", "price {v:.0f}% above its 200-day avg"),
    "Price_to_MA50_%": ("price {v:.0f}% below its 50-day avg", "price {v:.0f}% above its 50-day avg"),
    "Price_to_MA20_%": ("price {v:.0f}% below its 20-day avg", "price {v:.0f}% above its 20-day avg"),
    "MA50_to_MA200_%": ("50-day avg {v:.0f}% below the 200-day avg", "50-day avg {v:.0f}% above the 200-day avg"),
    "MA20_to_MA50_%": ("20-day avg {v:.0f}% below the 50-day avg", "20-day avg {v:.0f}% above the 50-day avg"),
    "Return_5d_%": ("5-day return {v:.1f}%", "5-day return {v:.1f}%"),
    "Return_10d_%": ("10-day return {v:.1f}%", "10-day return {v:.1f}%"),
    "Return_20d_%": ("20-day return {v:.1f}%", "20-day return {v:.1f}%"),
    "RSI_14": ("RSI {v:.0f} (oversold)", "RSI {v:.0f} (overbought)"),
    "Stoch_14": ("stochastic {v:.0f} (bottom of its range)", "stochastic {v:.0f} (top of its range)"),
    "MACD_Hist_%": ("MACD momentum {v:.2f}% (negative)", "MACD momentum {v:.2f}% (positive)"),
    "BB_Position": ("near its lower Bollinger band ({v:.2f})", "near its upper Bollinger band ({v:.2f})"),
    "Relative_Volume": ("volume {v:.2f}x its 20-day avg", "volume {v:.2f}x its 20-day avg"),
    "Volatility_30d": ("30-day volatility {v:.2f}%", "30-day volatility {v:.2f}%"),
    "Stock_Beta_Proxy": ("beta vs market {v:.2f}", "beta vs market {v:.2f}"),
    "Alpha_5d": ("{v:.1f}% behind the market this week", "{v:.1f}% ahead of the market this week"),
    "Alpha_MA5": ("lagging the market by {v:.2f}%/day", "beating the market by {v:.2f}%/day"),
    "Vol_Adjusted_Return_5d": ("small 5-day move for its volatility ({v:.2f})", "large 5-day move for its volatility ({v:.2f})"),
    "Stock_USD_Effect": ("pound move effect {v:.2f}", "pound move effect {v:.2f}"),
    "Stock_News_Response": ("moved against the news tone ({v:.3f})", "moved with the news tone ({v:.3f})"),
    "Volume_Driven_Sentiment": ("volume and news tone disagree ({v:.3f})", "volume and news tone agree ({v:.3f})"),
    "Stock_x_Market_Trend": ("weak trend in a weak market ({v:.2f})", "strong trend in a strong market ({v:.2f})"),
}


def short_phrase(feature, value, market_median):
    """Human phrase for one driver: 'price 19% below its 200-day avg'."""
    low, high = SHORT_PHRASES.get(feature, (None, None))
    if low is None:
        spec = FEATURE_INFO.get(feature, {}).get("fmt", "{:.2f}")
        return f"{feature_label(feature)} {_fmt(value, spec)}"
    tmpl = low if (pd.notna(value) and pd.notna(market_median) and value < market_median) else high
    try:
        return tmpl.format(v=abs(value))
    except (TypeError, ValueError):
        return f"{feature_label(feature)} {_fmt(value, '{:.2f}')}"


def feature_label(feature):
    return FEATURE_INFO.get(feature, {}).get("label", feature)


def market_wide_features(expl):
    """Features with the same value for every stock today (market context): they move all scores, not the ranking."""
    if expl is None or expl.empty:
        return []
    nun = expl.groupby("Feature")["Value"].nunique()
    return list(nun[nun <= 1].index)


def driver_table(expl, ticker, top=10):
    """What separates this stock from the other stocks today (market-wide factors removed)."""
    if expl is None or expl.empty:
        return pd.DataFrame()
    mine = expl[expl["Ticker"] == ticker].copy()
    if mine.empty:
        return pd.DataFrame()
    avg = expl.groupby("Feature")["Contribution"].mean()
    mine["Effect"] = mine["Contribution"] - mine["Feature"].map(avg)
    mine = mine[~mine["Feature"].isin(market_wide_features(expl))].copy()
    mine["Factor"] = mine["Feature"].map(feature_label)
    mine = mine.sort_values("Effect", key=lambda s: s.abs(), ascending=False)
    return mine.head(top).reset_index(drop=True)


def _fmt(value, spec):
    try:
        if value is None or pd.isna(value):
            return "-"
        return spec.format(value)
    except (TypeError, ValueError):
        return "-"


def driver_reading(row):
    """One plain sentence for a driver row: what the number is and why the model cared."""
    info = FEATURE_INFO.get(row["Feature"], {})
    label = info.get("label", row["Feature"])
    spec = info.get("fmt", "{:.2f}")
    value, median = row["Value"], row["Market_Median"]
    side = "low" if (pd.notna(value) and pd.notna(median) and value < median) else "high"
    meaning = info.get(side, "")
    text = f"**{label}** = {_fmt(value, spec)} (market median {_fmt(median, spec)})"
    if meaning:
        text += f": {meaning}"
    return text


def why_bullets(expl, ticker, n=4):
    """Short 'the model likes/dislikes this because...' lines for one stock."""
    d = driver_table(expl, ticker, top=n)
    if d.empty:
        return []
    out = []
    for _, r in d.iterrows():
        verb = "pushed the score up" if r["Effect"] > 0 else "pulled the score down"
        out.append(f"{driver_reading(r)}. This {verb}.")
    return out


def main_driver_map(expl, pred):
    """Ticker -> short text of the biggest stock-specific driver (for the ranking table)."""
    if expl is None or expl.empty or pred is None or pred.empty:
        return {}
    wide = set(market_wide_features(expl))
    avg = expl.groupby("Feature")["Contribution"].mean()
    e = expl[~expl["Feature"].isin(wide)].copy()
    e["Effect"] = e["Contribution"] - e["Feature"].map(avg)
    e["Abs"] = e["Effect"].abs()
    top = e.sort_values("Abs", ascending=False).groupby("Ticker").head(1)
    out = {}
    for _, r in top.iterrows():
        arrow = "\u2191" if r["Effect"] > 0 else "\u2193"
        out[r["Ticker"]] = f"{arrow} {short_phrase(r['Feature'], r['Value'], r['Market_Median'])}"
    return out


def rating_cutoffs(pred):
    """Score range each rating covers today (the ratings are relative: top 20% of today's list, etc.)."""
    if pred is None or pred.empty:
        return pd.DataFrame()
    g = pred.groupby("Rating")
    out = pd.DataFrame({"Stocks": g.size(), "Min score": g["Score"].min(), "Max score": g["Score"].max()})
    return out.reindex(RATINGS).dropna(how="all")


def driver_frame(expl, ticker, top=10):
    """Display-ready driver table for one stock: factor, its value, the market median, position, effect."""
    d = driver_table(expl, ticker, top=top)
    if d.empty:
        return pd.DataFrame()
    rows = []
    for _, r in d.iterrows():
        spec = FEATURE_INFO.get(r["Feature"], {}).get("fmt", "{:.2f}")
        rows.append({
            "Factor": r["Factor"],
            "This stock": _fmt(r["Value"], spec),
            "Market median": _fmt(r["Market_Median"], spec),
            "Position in market": f"{r['Value_Pct_Rank']:.0%}",
            "Effect": float(r["Effect"]),
            "Direction": "helps" if r["Effect"] > 0 else "hurts",
        })
    return pd.DataFrame(rows)


def market_backdrop(expl):
    """How much today's market-wide context moves every stock's score (same for all, so it never changes the rank)."""
    if expl is None or expl.empty:
        return None
    wide = market_wide_features(expl)
    if not wide:
        return None
    per_stock = expl[expl["Feature"].isin(wide)].groupby("Ticker")["Contribution"].sum()
    return {"effect": float(per_stock.median()), "low": float(per_stock.min()), "high": float(per_stock.max()),
            "features": [feature_label(f) for f in wide]}
